# Paths
IMAGE_DIR = "crops/Crop_224_5fp_100K"
TRAIN_CSV = "DataChallenge2026/occlusion_datasets/train.csv"
TEST_CSV = "DataChallenge2026/occlusion_datasets/test_students.csv"
OUTPUT_PRED = "DataChallenge2026/out/test_predictions.csv"

# Dataset split
VAL_SIZE = 20000

# Training parameters
BATCH_SIZE = 64
NUM_WORKERS = 4
LEARNING_RATE = 0.001
NUM_EPOCHS = 1

# Model parameters
MODEL_NAME = "efficientnet_b0" # "mobilenet_v3_small"
NUM_CLASSES = 1

# Device
USE_CUDA = True
DEVICE = None  # Will be set in main.py