import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch, Gain

# === CONFIG ===
SAMPLE_RATE = 8000
TARGET_DURATION = 3  # seconds
TARGET_SAMPLES = SAMPLE_RATE * TARGET_DURATION

input_root = "data/train_data"          # your dataset folder
output_root = "data/processed_3sec"     # new processed folder

# Augmentation pipeline (mild for baby cries)
augment = Compose([
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    TimeStretch(min_rate=0.85, max_rate=1.15, p=0.5),
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5),
    Gain(min_gain_db=-6, max_gain_db=6, p=0.5)
])

# Make output dirs
os.makedirs(output_root, exist_ok=True)

# --- 1. Load all files + find class counts ---
class_counts = {}
file_paths = {}

for cls in os.listdir(input_root):
    class_folder = os.path.join(input_root, cls)
    if not os.path.isdir(class_folder):
        continue
    files = [os.path.join(class_folder, f) for f in os.listdir(class_folder) if f.endswith(".wav")]
    file_paths[cls] = files
    class_counts[cls] = len(files)

print("\n=== Original Class Counts ===")
for c, n in class_counts.items():
    print(f" {c}: \t{n}")

max_count = max(class_counts.values())

# --- 2. Function to trim/pad to 3 sec ---
def fix_length(audio):
    if len(audio) > TARGET_SAMPLES:
        return audio[:TARGET_SAMPLES]
    elif len(audio) < TARGET_SAMPLES:
        pad = TARGET_SAMPLES - len(audio)
        return np.pad(audio, (0, pad))
    return audio

# --- 3. Process + augment to balance ---
for cls, files in file_paths.items():
    out_folder = os.path.join(output_root, cls)
    os.makedirs(out_folder, exist_ok=True)

    print(f"\nProcessing class: {cls}")
    idx = 0

    # Save original files (trim/pad)
    for path in tqdm(files):
        audio, sr = librosa.load(path, sr=SAMPLE_RATE)
        audio = fix_length(audio)
        sf.write(os.path.join(out_folder, f"{cls}_{idx}.wav"), audio, SAMPLE_RATE)
        idx += 1

    # Add augmented samples until balanced
    needed = max_count - len(files)
    print(f"Augmenting {needed} extra samples for {cls}...")

    for i in range(needed):
        audio, _ = librosa.load(np.random.choice(files), sr=SAMPLE_RATE)
        audio = fix_length(audio)
        aug_audio = augment(samples=audio, sample_rate=SAMPLE_RATE)
        sf.write(os.path.join(out_folder, f"{cls}_aug_{i}.wav"), aug_audio, SAMPLE_RATE)
