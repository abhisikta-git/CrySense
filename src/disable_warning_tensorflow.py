import os

# Suppress most TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # 0 = all logs, 1 = info, 2 = warning, 3 = error only

# Optional: prevent TF from pre-allocating all GPU memory (safer)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf

# Check what devices are visible
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\nGPU available: {len(gpus)} detected")
    for gpu in gpus:
        print(" -> ", gpu.name)
else:
    print("\nNo GPU detected — using CPU instead.")
