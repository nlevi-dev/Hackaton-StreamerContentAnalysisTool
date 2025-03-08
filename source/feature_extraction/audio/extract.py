import os
import sys
import re
import numpy as np
import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForSequenceClassification

if len(sys.argv) < 2:
    print("Usage: extract.py path_to_mp3")
    sys.exit(1)
path = sys.argv[1]

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
    audio, _ = librosa.load(input_file, sr=44100)
    audio_16k = librosa.resample(audio, orig_sr=44100, target_sr=16000)
    
    # Split audio into 10-minute segments
    segment_duration = 10 * 60  # 10 minutes in seconds
    duration_seconds = int(librosa.get_duration(y=audio, sr=44100))
    num_segments = duration_seconds // segment_duration
    
    with open("stage1.txt", "w") as log_file:
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
                    log_file.write(f"Word: '{word}' | Start: {adjusted_start_time:.2f}s | End: {adjusted_end_time:.2f}s\n")
            else:
                log_file.write("No timestamps available.\n")
            
            # Update cumulative offset by the segment duration
            cumulative_offset += segment_duration

def to_sentences(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    sentences = []
    current_sentence = []
    start_time = None
    end_time = None

    for line in lines:
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
    with open('stage2.txt', 'a') as output_file:
        for start, end, sentence in sentences:
            output_file.write(f"[{start:.2f} - {end:.2f}] {sentence}\n")


model_name = "tabularisai/multilingual-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to("cuda")

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return torch.argmax(probabilities, dim=-1).item()

def add_sentiment(file_path):
    # Read sentences from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Open the output file to write the results
    with open('stage3.txt', 'w') as output_file:
        # Process each line to predict sentiment and append it
        result = []
        for line in lines:
            # Extract the sentence part (assuming timecode is at the start)
            timecode, sentence = line.split(']', 1)
            sentence = sentence.strip()
            start, end = timecode[1:].split(' - ')
            start = int(float(start))
            end = int(float(end))
            sentiment = predict_sentiment(sentence)
            result.append([start,end,sentiment,sentence])
    return np.array(result)

print("stage1")
transcribe_audio(path)
print("stage2")
to_sentences("stage1.txt")
print("stage3")
result = add_sentiment("stage2.txt")
np.save(path[:-4]+"_audio_raw.npy",result)
os.remove("stage1.txt")
os.remove("stage2.txt")