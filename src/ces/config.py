
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "DUTS"
PROCESSED_DATA_DIR = DATA_DIR / "processed" / "DUTS"
SPLIT_DIR = DATA_DIR / "splits"

CAPTURED_DIR = ROOT_DIR / "captured"

MODEL_DIR = ROOT_DIR / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
FINAL_MODEL_DIR = MODEL_DIR / "final"

OUTPUT_DIR = ROOT_DIR / "outputs"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"
LOG_DIR = OUTPUT_DIR / "logs"

REPORT_DIR = ROOT_DIR / "reports"


# ============================================================
# DATASET CONFIGURATION
# ============================================================

IMAGE_SIZE = 256

# Selected model: Experiment A (128 x 128).
# Input images are resized to this resolution during inference.
MODEL_INPUT_SIZE = 128

# Ground-truth masks are binarized using pixel values >= 128.
MASK_THRESHOLD = 128

TRAIN_RATIO = 0.90

RANDOM_SEED = 42


# ============================================================
# MODEL CONFIGURATION
# ============================================================

INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 1

BASE_CHANNELS = 16

# Bilinear interpolation is used for decoder upsampling.
ALIGN_CORNERS = False


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 16

EPOCHS = 40

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

NUM_WORKERS = 2

USE_DICE_LOSS = True

DICE_WEIGHT = 0.5

CHECKPOINT_INTERVAL = 5


# ============================================================
# INFERENCE CONFIGURATION
# ============================================================

# Binary prediction threshold applied after sigmoid.
THRESHOLD = 0.50

# CPU-only inference with one thread.
CPU_THREADS = 1

WARMUP_RUNS = 1

BENCHMARK_RUNS = 20

MAX_CPU_TIME_MS = 50.0

MAX_MODEL_SIZE_BYTES = 4 * 1024 * 1024


# ============================================================
# EVALUATION CONFIGURATION
# ============================================================

BOUNDARY_TOLERANCE = 2

IOU_OBJECT_WEIGHT = 0.5
IOU_BACKGROUND_WEIGHT = 0.5

QUALITY_IOU_WEIGHT = 0.70
QUALITY_BOUNDARY_WEIGHT = 0.30


# ============================================================
# MODEL FILE
# ============================================================

# Official selected model.
FINAL_MODEL_PATH = FINAL_MODEL_DIR / "swiftmask_u2_v2.pt"


# ============================================================
# RANDOM SEED
# ============================================================

def set_random_seed(seed=RANDOM_SEED):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
