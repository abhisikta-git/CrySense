import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

from pathlib import Path

VIS_DIR = Path("./outputs/visualizations/")
VIS_DIR.mkdir(parents=True, exist_ok=True)

def visualize_audio(audio_file_path, save_path=None):
    """Visualize audio waveform, spectrogram, and MFCC"""
    audio_data, sample_rate = librosa.load(audio_file_path)
    
    plt.figure(figsize=(14, 8))
    
    # Waveform
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(audio_data, sr=sample_rate)
    plt.title('Waveform')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    # Spectrogram
    plt.subplot(3, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    
    # MFCC
    plt.subplot(3, 1, 3)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    librosa.display.specshow(mfcc, sr=sample_rate, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    
    plt.tight_layout()
    
    # Generate save path from input filename
    if save_path is None:
        input_path = Path(audio_file_path)
        save_path = VIS_DIR / f"{input_path.stem}.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    
    plt.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <audio_file_path>")
    else:
        visualize_audio(sys.argv[1])