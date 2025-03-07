import re

def process_speech_to_text(file_path):
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
    with open('processed_sentences.txt', 'a') as output_file:
        for start, end, sentence in sentences:
            output_file.write(f"[{start:.2f} - {end:.2f}] {sentence}\n")

if __name__ == "__main__":
    process_speech_to_text('/home/andris/ByborgAI/source/feature_extraction/speech_to_text/speech_to_text_log_accel.txt')