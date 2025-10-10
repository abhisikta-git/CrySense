import ffmpeg
from pathlib import Path
from typing import Optional, List

from utils import log

# ============================================================================
# Configuration
# ============================================================================
OUTPUT_DIR = Path("data/preprocessed_audio")
LOG_FILE = Path("outputs/logs/preprocess_log.txt")

SAMPLE_RATE = 16000
CHANNELS = 1

# Audio filter thresholds - tuned for baby cry ML preprocessing
SILENCE_THRESHOLD_DB = "-50dB"  # Lower threshold to preserve quiet sounds
SILENCE_DURATION_SEC = 2.0      # Only trim long silences (2+ seconds)
HIGHPASS_FREQ_HZ = 200          # Remove rumble while keeping cry fundamentals
LOWPASS_FREQ_HZ = 8000          # Preserve upper harmonics of baby cries


# ============================================================================
# Directory Setup
# ============================================================================
def ensure_directories():
    """Create output and log directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Audio Validation
# ============================================================================
def has_audio_stream(file_path: Path) -> bool:
    """
    Check if a file contains at least one valid audio stream.
    
    Uses ffprobe to inspect the file's streams without decoding.
    Returns False if the file is corrupted or not a media file.
    """
    try:
        probe = ffmpeg.probe(str(file_path))
        audio_streams = [
            stream for stream in probe['streams'] 
            if stream['codec_type'] == 'audio'
        ]
        return len(audio_streams) > 0
    except Exception as e:
        log(LOG_FILE, f"Invalid audio file {file_path.name}: {e}")
        return False


# ============================================================================
# Path Management
# ============================================================================
def calculate_output_path(
    input_path: Path, 
    base_input_dir: Optional[Path] = None
) -> Path:
    """
    Determine output path, preserving folder structure for batch processing.
    
    If base_input_dir is provided and input_path is within it, the relative
    directory structure is preserved under OUTPUT_DIR.
    """
    if base_input_dir and input_path.is_relative_to(base_input_dir):
        relative_subdir = input_path.parent.relative_to(base_input_dir)
        output_dir = OUTPUT_DIR / relative_subdir
    else:
        output_dir = OUTPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use stem to get filename without extension
    return output_dir / f"{input_path.stem}.wav"


# ============================================================================
# FFmpeg Filter Chain
# ============================================================================
def apply_audio_filters(stream):
    """
    Apply audio cleaning filters optimized for baby cry ML preprocessing.
    
    Pipeline:
    1. Silence removal - only trims excessive silence (2+ sec at start)
       Does NOT trim end to preserve natural cry patterns
    2. Dynamic normalization - equalizes volume while preserving dynamics
    3. High-pass filter - removes low-frequency rumble below 200Hz
    4. Low-pass filter - removes noise above 8kHz while keeping cry harmonics
    
    Note: Cry intensity and timing patterns are preserved as ML features
    """
    return (
        stream
        .filter(
            "silenceremove",
            start_periods=1,
            start_threshold=SILENCE_THRESHOLD_DB,
            start_silence=SILENCE_DURATION_SEC,
            # No stop_periods - preserve natural ending and pauses
        )
        .filter("dynaudnorm")
        .filter("highpass", f=HIGHPASS_FREQ_HZ)
        .filter("lowpass", f=LOWPASS_FREQ_HZ)
    )


def execute_ffmpeg_conversion(input_path: Path, output_path: Path) -> bool:
    """
    Run ffmpeg to convert and filter audio file.
    
    Returns True if conversion succeeded, False otherwise.
    Logs any errors encountered during processing.
    """
    try:
        stream = ffmpeg.input(str(input_path))
        stream = apply_audio_filters(stream)
        
        (
            stream
            .output(
                str(output_path),
                acodec="pcm_s16le",  # 16-bit PCM
                ac=CHANNELS,          # Mono
                ar=SAMPLE_RATE,       # 16kHz
                f="wav"
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return True
        
    except ffmpeg.Error as e:
        log(LOG_FILE, f"Error processing {input_path.name}")
        if e.stderr:
            error_msg = e.stderr.decode("utf-8", errors="ignore")
            log(LOG_FILE, f"FFmpeg error: {error_msg}")
            print(f"FFmpeg error for {input_path.name}:")
            print(error_msg)
        return False


# ============================================================================
# Output Verification
# ============================================================================
def verify_output(output_path: Path, input_name: str) -> bool:
    """
    Verify that the output file was created and has content.
    
    Logs file size if successful, logs failure if empty or missing.
    """
    if output_path.exists() and output_path.stat().st_size > 0:
        file_size = output_path.stat().st_size
        log(LOG_FILE, f"Saved: {output_path} ({file_size:,} bytes)")
        return True
    else:
        log(LOG_FILE, f"Failed: Empty output for {input_name}")
        return False


# ============================================================================
# Single File Processing
# ============================================================================
def preprocess_file(
    input_path: Path, 
    base_input_dir: Optional[Path] = None
) -> Optional[Path]:
    """
    Preprocess a single audio file and return path to the processed .wav.
    
    Returns None if processing fails at any stage.
    """
    ensure_directories()
    
    # Validate input file exists
    if not input_path.exists():
        log(LOG_FILE, f"File not found: {input_path}")
        return None
    
    # Verify it's a valid audio file
    if not has_audio_stream(input_path):
        log(LOG_FILE, f"Skipping {input_path.name} - not a valid audio file")
        return None
    
    # Calculate output path
    output_path = calculate_output_path(input_path, base_input_dir)
    
    log(LOG_FILE, f"Processing {input_path.name} -> {output_path}")
    
    # Execute ffmpeg conversion
    if not execute_ffmpeg_conversion(input_path, output_path):
        return None
    
    # Verify output
    if verify_output(output_path, input_path.name):
        return output_path
    else:
        return None


# ============================================================================
# Batch Processing
# ============================================================================
def collect_audio_files(directory: Path) -> List[Path]:
    """Recursively collect all files from a directory."""
    return [file for file in directory.rglob("*") if file.is_file()]


def preprocess_directory(input_dir: Path) -> List[Path]:
    """Process all audio files in a directory recursively."""
    processed_files = []
    audio_files = collect_audio_files(input_dir)
    
    log(LOG_FILE, f"Found {len(audio_files)} files in {input_dir}")
    
    for file in audio_files:
        output = preprocess_file(file, base_input_dir=input_dir)
        if output:
            processed_files.append(output)
    
    return processed_files


# ============================================================================
# Main Entry Point
# ============================================================================
def preprocess(input_target: str | Path) -> List[Path]:
    """
    Preprocess audio files from a file or directory.
    
    Args:
        input_target: Path to a single audio file or directory
        
    Returns:
        List of paths to successfully processed .wav files
    """
    input_target = Path(input_target)
    processed_files = []
    
    if not input_target.exists():
        log(LOG_FILE, f"Invalid input path: {input_target}")
        return processed_files
    
    if input_target.is_file():
        output = preprocess_file(input_target)
        if output:
            processed_files.append(output)
    
    elif input_target.is_dir():
        processed_files = preprocess_directory(input_target)
    
    log(LOG_FILE, f"Finished preprocessing {len(processed_files)} file(s).")
    return processed_files


# ============================================================================
# CLI Interface
# ============================================================================
def print_usage():
    """Print command-line usage instructions."""
    print("Usage:")
    print("  python preprocess.py <file_or_folder_path>")
    print("\nExamples:")
    print("  python preprocess.py data/raw_audio/rec1.mp3")
    print("  python preprocess.py data/raw_audio/rec1.m4a")
    print("  python preprocess.py data/raw_audio/")


def main():
    """CLI entry point."""
    import sys
    
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    input_path = sys.argv[1]
    processed = preprocess(input_path)
    
    print(f"\nSuccessfully processed {len(processed)} file(s)")
    if processed:
        print("\nOutput files:")
        for path in processed:
            print(f"  - {path}")


if __name__ == "__main__":
    main()