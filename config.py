import os

# Data paths
DATA_DIR = "data/msd_hippocampus"
DOWNLOAD_URL = "https://msd-for-human-interpretation.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar"
PROCESSED_DATA_DIR = "data/processed"

# Training Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
NUM_WORKERS = 4
TRAIN_SPLIT = 0.8

# Model Hyperparameters
IN_CHANNELS = 1
NUM_CLASSES = 1 # Background vs Hippocampus (binary segmentation)
