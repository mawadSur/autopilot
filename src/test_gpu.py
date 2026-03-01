import os
import sys

if os.getenv("DISABLE_GPU") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("All physical devices:", tf.config.list_physical_devices())

try:
    gpus = tf.config.list_physical_devices("GPU")
    print("Num GPUs Available:", len(gpus))
    for gpu in gpus:
        print("GPU device:", gpu)
except Exception as exc:
    print("CUDA init failed while listing GPUs.")
    print("Error:", exc)
    print("Hints:")
    print("- Ensure you are running under WSL2 (not WSL1) if using WSL.")
    print("- Verify the NVIDIA Windows driver is WSL-compatible and up to date.")
    print("- Check that /dev/dxg exists in WSL (GPU bridge).")
    print("- Confirm your TensorFlow build supports your OS/driver combo.")

if os.getenv("REQUIRE_GPU") == "1" and len(tf.config.list_physical_devices("GPU")) == 0:
    print("GPU required but none available. Exiting with code 1.")
    sys.exit(1)
