import torch
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import librosa
import numpy as np
import time

model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
model = AutoModelForAudioClassification.from_pretrained(model_name)
if torch.cuda.is_available():
    model.to("cuda")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

# Compile the model for optimized inference
model = torch.compile(model)

def predict_emotion_batch(audio_segments, device):
    # Process a batch of audio segments
    inputs = feature_extractor(audio_segments, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Predict emotions using inference mode
    with torch.inference_mode():
        logits = model(**inputs).logits
    
    # Get predicted emotions
    predicted_ids = torch.argmax(logits, dim=-1)
    emotions = [model.config.id2label[id.item()] for id in predicted_ids]
    
    return emotions

if __name__ == "__main__":
    audio_file = "/mnt-persist/test/1/Our_New_4500_Workstation_PCs_for_Editing.mp3"
    
    # Load the entire audio file once
    print("Loading audio file...")
    audio, sr = librosa.load(audio_file, sr=16000)
    audio_duration = len(audio) / sr
    segment_length = 10 * sr  # 10 seconds * sample rate
    num_segments = int(len(audio) // segment_length)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    with open("emotion_log.txt", "w") as log_file:
        # Process segments in batches
        batch_size = 12
        num_batches = num_segments // batch_size + (1 if num_segments % batch_size else 0)
        
        start_time = time.time()
        for batch in range(num_batches):
            
            # Calculate how many segments to process in this batch
            segments_in_batch = min(batch_size, num_segments - batch * batch_size)
            batch_segments = []
            time_indices = []
            
            # Prepare segments for this batch
            for i in range(segments_in_batch):
                segment_idx = batch * batch_size + i
                start_sample = segment_idx * segment_length
                end_sample = start_sample + segment_length
                segment = audio[start_sample:end_sample]
                batch_segments.append(segment)
                time_indices.append((segment_idx * 10, (segment_idx + 1) * 10))
            
            # Process the batch
            emotions = predict_emotion_batch(batch_segments, device)
            
            # Log results
            for (start_sec, end_sec), emotion in zip(time_indices, emotions):
                log_file.write(f"The predicted emotion from {start_sec} to {end_sec} seconds is: {emotion}\n")
                print(f"The predicted emotion from {start_sec} to {end_sec} seconds is: {emotion}")
            
            end_time = time.time()
            processing_time = end_time - start_time
            log_file.write(f"Time taken for batch prediction: {processing_time:.2f} seconds\n")
            print(f"Time taken for batch prediction: {processing_time:.2f} seconds")
        print(f"Total time taken for all predictions: {time.time() - start_time:.2f} seconds")
