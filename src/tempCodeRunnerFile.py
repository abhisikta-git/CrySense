import librosa

audio_path = "data/preprocessed_audio/4.wav"
y, sr = librosa.load(audio_path, sr=None)  # sr=None keeps original sample rate

print("Sample rate:", sr)
print("Duration (seconds):", librosa.get_duration(y=y, sr=sr))
