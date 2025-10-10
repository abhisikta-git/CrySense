import librosa
import numpy as np
from pathlib import Path
from datetime import datetime

def features_extractor(file_path):
    """Extract MFCC features from an audio file"""
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        
        if sample_rate != 22050:
            import scipy.signal as sps
            number_of_samples = round(len(audio) * float(22050) / sample_rate)
            audio = sps.resample(audio, number_of_samples)
            sample_rate = 22050
        
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        return mfccs_scaled_features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def log(LOG_FILE: Path, msg: str):
    print(msg)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg} \n")