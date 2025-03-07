from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True, 
    #use_flash_attn=True,  
    attn_implementation="eager"
)

# Ensure the model is moved to GPU after initialization
if device.startswith("cuda"):
    model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps='word',  # Set to 'word' for word-level timestamps
)


import librosa

def transcribe_audio(input_file, duration_minutes=2):
    duration_seconds = duration_minutes * 60
    audio, _ = librosa.load(input_file, sr=44100, duration=duration_seconds)
    audio_16k = librosa.resample(audio, orig_sr=44100, target_sr=16000)
    
    # Split audio into 10-minute segments
    segment_duration = 10 * 60  # 10 minutes in seconds
    num_segments = duration_seconds // segment_duration
    
    with open("speech_to_text_log_accel.txt", "w") as log_file:
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
                    # Adjust timestamps by adding the cumulative offset
                    adjusted_start_time = start_time + cumulative_offset
                    adjusted_end_time = end_time + cumulative_offset
                    log_file.write(f"Word: '{word}' | Start: {adjusted_start_time:.2f}s | End: {adjusted_end_time:.2f}s\n")
            else:
                log_file.write("No timestamps available.\n")
            
            # Update cumulative offset by the segment duration
            cumulative_offset += segment_duration

if __name__ == "__main__":
    transcribe_audio("/mnt-persist/test/1/Our_New_4500_Workstation_PCs_for_Editing.mp3", duration_minutes=130)
