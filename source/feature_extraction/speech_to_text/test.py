import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import soundfile as sf
import librosa
def transcribe_audio(input_file):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,
        #language="en",
    )

    # Load the audio file with a single channel
    audio_input, sample_rate = sf.read(input_file, always_2d=True)
    audio_input = audio_input.mean(axis=1)  # Convert to single channel by averaging

    # Resample to 44.1kHz if necessary
    if sample_rate != 44100:
        audio_input = librosa.resample(audio_input, orig_sr=sample_rate, target_sr=44100)

    # Transcribe the audio in chunks
    chunk_size = 30 * 44100  # 30 seconds per chunk at 44.1kHz sample rate
    num_segments = 5
    total_length = len(audio_input)
    transcribed_text_with_timestamps = ""
    current_time = 0.0

    for start in range(0, total_length, chunk_size * num_segments):
        end = min(start + chunk_size * num_segments, total_length)
        audio_chunk = audio_input[start:end]

        # Transcribe the audio chunk
        result = pipe(audio_chunk)

        # Extract text with timestamps for each segment
        for segment in result["chunks"]:
            start_time = current_time + segment["timestamp"][0]
            end_time = current_time + segment["timestamp"][1]
            text = segment["text"]
            transcribed_text_with_timestamps += f"[{start_time:.2f}-{end_time:.2f}] {text} "

        # Update current time for the next chunk
        current_time += (end - start) / 44100.0

        # Write to log file after processing each set of segments
        with open("speech_to_text_log.txt", "a") as log_file:
            log_file.write(transcribed_text_with_timestamps.strip() + "\n")
        
        # Clear the transcribed text for the next set of segments
        transcribed_text_with_timestamps = ""

    return transcribed_text_with_timestamps.strip()

if __name__ == "__main__":
    transcribe_audio("Our New _4500 Workstation PCs for Editing [G4JoDcsk62A].mp3") # filepath should be here 