# Paths
IMAGE_DIR = "crops/Crop_224_5fp_100K"
TRAIN_CSV = "DataChallenge2026/occlusion_datasets/train.csv"
TEST_CSV = "DataChallenge2026/occlusion_datasets/test_students.csv"
OUTPUT_PRED = "DataChallenge2026/out/test_predictions2.csv"

# Dataset split
VAL_SIZE = 20000

# Training parameters
BATCH_SIZE = 16
NUM_WORKERS = 4
LEARNING_RATE = 3e-4 # 0.001
NUM_EPOCHS = 20
LR = 1e-4
PATIENCE = 5

# Model parameters
MODEL_NAME = "efficientnet_b2" # "efficientnet_b0" / "mobilenet_v3_small"
NUM_CLASSES = 1

# Device
USE_CUDA = True
