import librosa, os

folder = "data/train_data/discomfort"

for f in os.listdir(folder):
    if f.endswith(".wav"):
        y, sr = librosa.load(os.path.join(folder, f), sr=None)
        print(f, " -> Sample Rate:", sr)
