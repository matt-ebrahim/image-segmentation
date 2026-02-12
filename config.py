import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_NAME = "Task06_Lung"
RAW_DATA_DIR = os.path.join(DATA_DIR, DATASET_NAME)
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Data paths
IMAGES_TR_DIR = os.path.join(RAW_DATA_DIR, "imagesTr")
LABELS_TR_DIR = os.path.join(RAW_DATA_DIR, "labelsTr")
IMAGES_TS_DIR = os.path.join(RAW_DATA_DIR, "imagesTs")

# Download settings
DOWNLOAD_URL = "https://msd-for-human-interpretation.s3-us-west-2.amazonaws.com/Task06_Lung.tar"
DATASET_TAR_FILE = os.path.join(DATA_DIR, "Task06_Lung.tar")

# Training Hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
NUM_WORKERS = 4
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Model Hyperparameters
IN_CHANNELS = 1
NUM_CLASSES = 2  # Background + Lung tumor
BASE_CHANNEL_COUNT = 32

# Mixed Precision Training
USE_AMP = True

# Early Stopping
EARLY_STOPPING_PATIENCE = 15

# Learning Rate Scheduler
LR_SCHEDULER_T_MAX = NUM_EPOCHS

# Loss weights
DICE_LOSS_WEIGHT = 0.5
CE_LOSS_WEIGHT = 0.5

# Device
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") or os.path.exists("/proc/driver/nvidia") else "cpu"

# Logging
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
VISUALIZATION_DIR = os.path.join(PROJECT_ROOT, "visualizations")

# Data Augmentation
AUGMENTATION_PROBABILITY = 0.5