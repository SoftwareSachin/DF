"""
Configuration settings for the deep fake detection system.
"""
import os

# Model paths and settings
MESONET_WEIGHTS_PATH = "weights/mesonet_weights.h5"
XCEPTION_WEIGHTS_PATH = "weights/xception_deepfake_weights.h5"

# Detection thresholds
DEEPFAKE_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.7

# Video processing settings
MAX_FRAMES_TO_ANALYZE = 30
FRAME_SKIP_INTERVAL = 5

# Supported file formats
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

# Face detection settings
MIN_FACE_SIZE = 80
FACE_CONFIDENCE_THRESHOLD = 0.9

# Frequency analysis settings
FFT_WINDOW_SIZE = 64
DCT_BLOCK_SIZE = 8

# Model input dimensions
MESONET_INPUT_SIZE = (256, 256)
XCEPTION_INPUT_SIZE = (299, 299)

# Environment variables
MODELS_DIR = os.getenv("MODELS_DIR", "models")
TEMP_DIR = os.getenv("TEMP_DIR", "temp")
