import os
import sys
import re
import _pickle as pickle
import numpy as np
import pandas as pd
import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

if len(sys.argv) < 2:
    print("Usage: extract.py path_to_mp3 [--debug]")
    sys.exit(1)
path = sys.argv[1]
debug = "--debug" in sys.argv

model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    attn_implementation="eager",
)
model.to("cuda")
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float16,
    device="cuda",
    return_timestamps="word",
)

def transcribe_audio(input_file):
    """Transcribe audio from a file and return a log of words with timestamps.

    This function processes an audio file by loading it, resampling it to a target
    sample rate, and splitting it into 10-minute segments. Each segment is then
    transcribed using a speech recognition pipeline, and the transcriptions are
    logged with their respective start and end timestamps.

    Args:
        input_file (str): The path to the audio file to be transcribed.

    Returns:
        str: A log file as a string, where each line contains a word from the
        transcription along with its start and end timestamps in the format:
        "Word: '<word>' | Start: <start_time>s | End: <end_time>s".

    Example:
        Given an audio file, the function will return a string like:
        "Word: 'Hello' | Start: 0.00s | End: 0.50s\n
         Word: 'world' | Start: 0.51s | End: 1.00s\n
         Word: '!' | Start: 1.01s | End: 1.10s\n"

    Notes:
        - The audio is resampled to 16 kHz before processing.
        - The function handles audio files of arbitrary length by processing
          them in 10-minute segments.
        - If no timestamps are available for a segment, a message is logged
          indicating the absence of timestamps.
    """
    audio, _ = librosa.load(input_file, sr=44100)
    audio_16k = librosa.resample(audio, orig_sr=44100, target_sr=16000)
    
    # Split audio into 10-minute segments
    segment_duration = 10 * 60  # 10 minutes in seconds
    duration_seconds = int(librosa.get_duration(y=audio, sr=44100))
    num_segments = duration_seconds // segment_duration

    log_file = ""
    cumulative_offset = 0  # Initialize cumulative offset for timestamps
    for i in range(num_segments):
        start = i * segment_duration
        end = start + segment_duration
        segment = audio_16k[start * 16000:end * 16000]  # 16000 is the target sample rate
        
        result = pipe(segment)
        
        if 'chunks' in result:
            for chunk in result['chunks']:
                word = chunk['text']
                start_time, end_time = chunk['timestamp']
                adjusted_start_time = start_time + cumulative_offset
                adjusted_end_time = end_time + cumulative_offset
                log_file += f"Word: '{word}' | Start: {adjusted_start_time:.2f}s | End: {adjusted_end_time:.2f}s\n"
        else:
            log_file += "No timestamps available.\n"
        
        # Update cumulative offset by the segment duration
        cumulative_offset += segment_duration
    return log_file

print("stage1")
log_file = transcribe_audio(path)
del pipe
del processor
del model

def to_sentences(file_path):
    """Converts a log file of transcribed words with timestamps into sentences.

    This function processes a string containing lines of transcribed words, each with
    associated start and end timestamps, and groups them into sentences. Sentences are
    determined based on sentence-ending punctuation marks (e.g., '.', '!', '?'). Each
    sentence is returned with its start and end timestamps.

    Args:
        file_path (str): A string representation of the file content, where each line
            contains a word with its start and end timestamps in the format:
            "Word: '<word>' | Start: <start_time>s | End: <end_time>s".

    Returns:
        str: A formatted string where each line represents a sentence with its start
        and end timestamps in the format:
        "[<start_time> - <end_time>] <sentence>"

    Example:
        Given a file content:
        "Word: 'Hello' | Start: 0.00s | End: 0.50s\n
         Word: 'world' | Start: 0.51s | End: 1.00s\n
         Word: '!' | Start: 1.01s | End: 1.10s\n"

        The function will return:
        "[0.00 - 1.10] Hello world !\n"
    """
    lines = file_path.split("\n")

    sentences = []
    current_sentence = []
    start_time = None
    end_time = None

    for line in lines:
        if len(line) == 0:
            continue
        match = re.match(r"Word: '(.+?)' \| Start: ([\d.]+)s \| End: ([\d.]+)s", line)
        if match:
            word, start, end = match.groups()
            start, end = float(start), float(end)

            if not current_sentence:
                start_time = start

            current_sentence.append(word)
            end_time = end

            # Check if the word ends with a sentence-ending punctuation
            if word.endswith(('.', '!', '?')) or (len(current_sentence) > 1 and current_sentence[-2].endswith(('.', '!', '?'))):
                sentences.append((start_time, end_time, ' '.join(current_sentence)))
                current_sentence = []

    # Handle any remaining sentence
    if current_sentence:
        sentences.append((start_time, end_time, ' '.join(current_sentence)))

    # Output the sentences in the desired format and append to processed_sentences.txt
    output_file = ""
    for start, end, sentence in sentences:
        output_file += f"[{start:.2f} - {end:.2f}] {sentence}\n"
    return output_file

print("stage2")
output_file = to_sentences(log_file)

model_name = "tabularisai/multilingual-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to("cuda")

def predict_sentiment(text):
    """Predicts the sentiment of a given text.

    This function uses a pre-trained sentiment analysis model to predict the sentiment
    of the input text. The text is tokenized and processed through the model, and the
    sentiment is determined based on the highest probability class.

    Args:
        text (str): The input text for which sentiment needs to be predicted.

    Returns:
        int: The predicted sentiment class index. Typically, this index corresponds to
        a specific sentiment category, such as positive, negative, or neutral.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return torch.argmax(probabilities, dim=-1).item()

def add_sentiment(file_path):
    """Adds sentiment analysis to each sentence in a file.

    This function processes a file containing sentences with timecodes, predicts the
    sentiment for each sentence, and returns an array with the start time, end time,
    sentiment, and the sentence itself.

    Args:
        file_path (str): The path to the file containing sentences with timecodes.

    Returns:
        np.ndarray: A numpy array where each row contains the start time, end time,
        predicted sentiment, and the sentence. The sentiment is represented as an
        integer index corresponding to a sentiment category.
    """
    lines = file_path.split("\n")

    result = []
    for line in lines:
        if len(line) == 0:
            continue
        # Extract the sentence part (assuming timecode is at the start)
        timecode, sentence = line.split(']', 1)
        sentence = sentence.strip()
        start, end = timecode[1:].split(' - ')
        start = int(float(start))
        end = int(float(end))
        sentiment = predict_sentiment(sentence)
        result.append([start, end, sentiment, sentence])
    return np.array(result)

print("stage3")
result = add_sentiment(output_file)
np.save(path[:-4]+"_audio_raw.npy",result)
del model
del tokenizer

CONTEXT_L = 0
CONTEXT_S = 40
STEP = 60
OFFSET = 30

raw = np.load(path[:-4]+"_audio_raw.npy")
END = int(raw[-1,1])

PROMPT_L = ""
PROMPT_S = ""
PROMPT = ""

chunks = []
for i in range(END//STEP):
    end_s = OFFSET+i*STEP+STEP
    start_s = end_s-CONTEXT_S
    end_l = start_s
    start_l = end_s-CONTEXT_L
    context_l = ""
    context_s = ""
    for s in raw:
        st = int(s[0])
        if start_l <= st and st < end_l:
            context_l += "\""+s[3]+"\"\n"
        elif start_s <= st and st <= end_s:
            context_s += s[3]+"\n"
    chunk = ""
    if len(context_l) > 0:
        chunk += PROMPT_L
        chunk += context_l
    chunk += PROMPT_S
    chunk += context_s
    chunk += PROMPT
    chunks.append(chunk)

def pickle_load(path):
    """Load an object from a pickle file.

    Args:
        path (str): The file path to the pickle file.

    Returns:
        object: The object loaded from the pickle file.
    """
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    return ret

def pickle_save(path, obj):
    """Save an object to a pickle file.

    Args:
        path (str): The file path where the object will be saved.
        obj (object): The object to be saved to the pickle file.
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def text_save(path, txt):
    """Save text to a file.

    Args:
        path (str): The file path where the text will be saved.
        txt (str): The text content to be written to the file.
    """
    with open(path, 'w') as f:
        f.write(txt)

def get_closest_idx(items, candidate):
    """Find the index of the item in a list that is closest to a candidate string.

    Uses the Jaro-Winkler distance metric to determine similarity.

    Args:
        items (list of str): A list of strings to compare against.
        candidate (str): The string to compare with the list items.

    Returns:
        int: The index of the item in the list that is most similar to the candidate.
    """
    m = 0
    ret_idx = 0
    idx = 0
    for item in items:
        dist = jaro.jaro_winkler_metric(item, candidate)
        if dist > m:
            m = dist
            ret_idx = idx
        idx += 1
    return ret_idx

def generate_onehot(items):
    """Generate a one-hot encoding prompt and a function to find the closest item.

    Constructs a prompt for selecting an item from a list and provides a function
    to find the closest item to a given input using the Jaro-Winkler distance.

    Args:
        items (list of str): A list of items to include in the prompt.

    Returns:
        tuple: A tuple containing the prompt string and a function to find the closest item.
    """
    prepromt = "Answer from the following list with only using a word from it [" + ",".join(items) + "]!"
    closest = lambda a: get_closest_idx(items, a)
    return (prepromt, closest)

preprompts = {
    "list[str]":("Answer with a short, comma separated list!",lambda a:[re.sub("[\\s]+","_",b.strip().lower()) for b in a.split(',') if b.strip() != '']),
    "list[int]":("Answer with a short, comma separated list!",lambda a:[int(re.sub("[^\\d]","",b)) for b in a.split(',') if len(b.strip())>0]),
    "list[float]":("Answer with a short, comma separated list!",lambda a:[float(re.sub("[^\\d\\.]","",b)) for b in a.split(',') if b.strip() != '']),
    "int":("Answer with a single integer number!",lambda a:int(re.sub("[^\\d]","",a)) if len(re.sub("[^\\d]","",a))>0 else 0),
    "float":("Answer with a single float number!",lambda a:float(re.sub("[^\\d\\.]","",a)) if len(re.sub("[^\\d\\.]","",a))>0 else 0.0),
    "bool":("Answer with a single yes or no!",lambda a:"y" in a),
}

# preprompts = {
#     "list[str]":("Answer with a short, comma separated list!",lambda a:a),
#     "list[int]":("Answer with a short, comma separated list!",lambda a:a),
#     "list[float]":("Answer with a short, comma separated list!",lambda a:a),
#     "int":("Answer with a single integer number!",lambda a:a),
#     "float":("Answer with a single float number!",lambda a:a),
#     "bool":("Answer with a single yes or no!",lambda a:a),
# }

def get_preprompt(key):
    if isinstance(key, str):
        return preprompts[key]
    if isinstance(key, list):
        return generate_onehot(key)
    raise Exception(":))))")

prompts = [
    # People & Speaking Patterns
    ("int", "How many times does a speaker interrupt another speaker?"),
    ("int", "How many times does a speaker use humor or sarcasm?"),
    ("int", "How many times do speakers address each other by name?"),
    ("int", "How many times do speakers address the audience directly?"),
    ("int", "How often do speakers talk over each other?"),
    ("int", "How frequently do speakers express excitement or enthusiasm?"),
    ("int", "How many rhetorical questions are asked in the transcript?"),
    # Topics & Technical Content
    ("int", "How many times is 'PC build' or 'building' mentioned?"),
    # ("list[str]", "Which specific computer components are mentioned (CPU, GPU, RAM, SSD, etc.)?"),
    # ("list[str]", "Which specific PC brands are mentioned (Intel, AMD, NVIDIA, ASUS, etc.)?"),
    ("int", "How many times do they mention benchmarks or performance comparisons?"),
    ("bool", "Did they mention troubleshooting or fixing an issue?"),
    ("int", "How often do they mention overclocking or optimization?"),
    # ("list[str]", "Which specific software tools or BIOS settings are mentioned?"),
    ("int", "How many times do they mention price or cost?"),
    ("int", "How often do they discuss PC aesthetics (RGB lighting, case design, etc.)?"),
    # Jokes, Reactions & Engagement Hooks
    ("int", "How many times does someone say something funny?"),
    ("int", "How often do speakers exaggerate for comedic effect?"),
    ("int", "How many times is a running joke referenced?"),
    ("int", "How many times does Linus make a self-deprecating joke?"),
    # ("list[str]", "What emotions do speakers express when reacting with surprise or frustration?"),
    ("int", "How often do speakers break the fourth wall (acknowledging the audience or video production)?"),
    # ("list[str]", "Which internet slang or memes do speakers use?"),
    ("int", "How often does Linus joke about dropping something?"),
    ("int", "How often do they make a reference to previous LTT videos?"),
    ("int", "How many times do they tease each other or engage in friendly banter?"),
    # Instructions & Step-by-Step Explanations
    ("int", "How many times do they use words like 'step,' 'next,' or 'now we'?"),
    ("int", "How many times does Linus give direct instructions?"),
    ("int", "How frequently do they explain the reasoning behind a step?"),
    ("int", "How many times do they mention safety precautions (ESD, handling delicate parts, etc.)?"),
    ("int", "How many times do they explain a concept in layman's terms?"),
    ("int", "How often do they reference 'best practices' or 'what you should do'?"),
    ("int", "How frequently do they correct themselves or change their approach?"),
    ("int", "How many times does a speaker express doubt about a step or decision?"),
    # Mistakes, Troubleshooting & Problem Solving
    ("int", "How many times does something go wrong in the build process?"),
    ("int", "How often do they acknowledge a mistake?"),
    ("int", "How many times do they try to troubleshoot an issue?"),
    ("int", "How frequently do they joke about things breaking or not working?"),
    ("int", "How many times do they mention something being more difficult than expected?"),
    ("int", "How often do they have to redo a step?"),
    # ("list[str]", "What last-minute changes do they make in the build?"),
    # ("list[str]", "What common mistakes do they mention viewers might make?"),
    ("int", "How often do they say 'we will fix it later' or something similar?"),
    # ("list[str]", "Which components or tools do they express frustration with?"),
    # Sponsorships, Branding & Call-to-Actions
    ("int", "How many times is a sponsor mentioned?"),
    ("int", "How many times does Linus explicitly read an ad or promotional message?"),
    # ("list[str]", "Which products are mentioned in a way that suggests sponsorship?"),
    ("int", "How often do they mention LTT store products?"),
    ("int", "How many times do they remind viewers to subscribe or like the video?"),
    ("int", "How frequently do they mention future videos or upcoming content?"),
    ("int", "How many times do they ask viewers for opinions in the comments?"),
    # ("list[str]", "Which external websites or resources do they mention?"),
    # ("list[str]", "Which other YouTube channels do they reference?"),
    ("int", "How many times do they promote their paid content (Floatplane, LTT Labs, etc.)?"),
]

torch.random.manual_seed(0)
model_path = "mistralai/Mistral-7B-Instruct-v0.3"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
# prompts = prompts[0:10]

def prompt(chunk, prompts):
    """Processes a transcript chunk with specified prompts to extract information.

    This function takes a chunk of transcript text and a list of prompts, then
    constructs messages for a text generation model to extract specific information
    from the transcript. The function uses a pre-defined pipeline to process these
    messages and returns the extracted results.

    Args:
        chunk (str): A segment of transcript text from a LinusTechTips video, typically
            involving computer building activities.
        prompts (list[tuple]): A list of tuples where each tuple contains:
            - A string representing the data type (e.g., "int").
            - A string describing the specific information to extract from the transcript.

    Returns:
        list: A list of extracted information corresponding to each prompt, processed
        by the text generation model.

    Example:
        >>> chunk = "In this step, we install the CPU cooler."
        >>> prompts = [("int", "How many times do they use words like 'step,' 'next,' or 'now we'?")]
        >>> prompt(chunk, prompts)
        [2]

    Notes:
        - The function assumes the use of a pre-trained language model for text generation.
        - The model is expected to be initialized and available as a global variable `pipe`.
    """
    messages = []
    pres = []
    for prompt in prompts:
        pre = get_preprompt(prompt[0])
        pres.append(pre[1])
        txt = ""
        txt += "Given the following transcript from a LinusTechTips video, where they are usually building computers, extract "
        txt += prompt[1]
        txt += "and "
        txt += pre[0] + "\n"
        txt += chunk + "\n"
        txt += pre[0] + " " + prompt[1]
        messages.append([
            {"role": "system", "content": "You are a data processing agent. You extract information from provided transcripts, which are wrapped in quotes."},
            {"role": "user", "content": txt},
        ])
    results = pipe(messages, batch_size=len(prompts), max_new_tokens=10)
    for i in range(len(results)):
        results[i] = pres[i](results[i][0]['generated_text'][-1]["content"])
        # print(prompts[i][1]+": "+str(results[i]))
    return results

idx = path.rfind("/")
pat = path[:idx]
nam = path[idx+1:-4]

os.makedirs(pat+"/feature_audio", exist_ok=True)
if debug:
    pd.set_option('display.width',os.get_terminal_size().columns)
    os.makedirs(pat+"/feature_audio_debug", exist_ok=True)
def ljust(s):
    """Left-justify the strings in a pandas Series.

    This function takes a pandas Series of strings, strips any leading or trailing
    whitespace from each string, and then left-justifies each string to the length
    of the longest string in the Series.

    Args:
        s (pd.Series): A pandas Series containing strings to be left-justified.

    Returns:
        pd.Series: A pandas Series with each string left-justified to the length
        of the longest string in the original Series.

    Example:
        >>> import pandas as pd
        >>> s = pd.Series(['apple', 'banana', 'cherry'])
        >>> ljust(s)
        0     apple
        1    banana
        2    cherry
        dtype: object

    Notes:
        - The function assumes that the input Series contains only string data.
        - Leading and trailing whitespace is removed from each string before
          left-justifying.
    """
    s = s.astype(str).str.strip()
    return s.str.ljust(s.str.len().max())

# chunks = chunks[50:52]

for i in range(len(chunks)):
    results = prompt(chunks[i], prompts)
    timestamp = i*STEP+STEP
    timestamp_formatted = f"{timestamp:06}"
    pickle_save(pat+"/feature_audio/"+timestamp_formatted+"_"+nam+".pkl",results)
    if debug:
        prs = np.array([[str(p[0]),str(p[1])] for p in prompts])
        prs = np.concatenate([prs,np.array([[str(r) for r in results]]).T],axis=1)
        df = pd.DataFrame(prs, columns=['datatype','prompt','value'])
        txt = df.apply(ljust).to_string(index=False,justify='left')
        text_save(pat+"/feature_audio_debug/"+timestamp_formatted+"_"+nam+".txt",txt)
        text_save(pat+"/feature_audio_debug/"+timestamp_formatted+"_"+nam+"_chunk.txt",chunks[i])
    print(str(i+1)+"/"+str(len(chunks)))