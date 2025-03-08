from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "tabularisai/multilingual-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
    return sentiment_map[torch.argmax(probabilities, dim=-1).item()]

# Read sentences from the file
with open('/home/ha61/andris/ByborgAI/source/feature_extraction/speech_to_text/processed_sentences.txt', 'r') as file:
    lines = file.readlines()

# Open the output file to write the results
with open('./transcript_with_sentiment.txt', 'w') as output_file:
    # Process each line to predict sentiment and append it
    for line in lines:
        # Extract the sentence part (assuming timecode is at the start)
        timecode, sentence = line.split(']', 1)
        sentence = sentence.strip()
        # Predict sentiment for the sentence
        sentiment = predict_sentiment(sentence)
        # Append sentiment between timecode and sentence
        new_line = f"{timecode}] [{sentiment}] [{sentence}]"
        output_file.write(new_line + '\n')
        print(new_line)
