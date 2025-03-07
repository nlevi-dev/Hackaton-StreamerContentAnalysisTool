from transformers import pipeline

pipe = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

audio_file = "path_to_audio.wav"
result = pipe(audio_file)

print(result)