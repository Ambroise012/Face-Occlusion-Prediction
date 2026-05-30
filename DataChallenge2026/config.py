# Paths
IMAGE_DIR = "crops/Crop_224_5fp_100K"
TRAIN_CSV = "DataChallenge2026/occlusion_datasets/train.csv"
TEST_CSV = "DataChallenge2026/occlusion_datasets/test_students.csv"
OUTPUT_PRED = "DataChallenge2026/out/test_predictions_MoE.csv"

# Training parameters
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS = 20
LR = 1e-4
PATIENCE = 5

# Model parameters
MODEL_NAME = [
    "convnext_tiny",
    "efficientnet_b2",
    "efficientnet_v2_s"
]
NUM_CLASSES = 1

# Device
USE_CUDA = True
