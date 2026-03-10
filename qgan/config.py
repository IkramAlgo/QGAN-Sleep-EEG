# qgan/config.py
# All settings in one place. Change anything here.

import os

# Data
EDF_FILE_PATH = "data/EPCTL03.edf"
EEG_CHANNEL   = 0
EPOCH_SECONDS = 30

# All available features in order — experiments use the first N of these
ALL_FEATURE_NAMES = ["Mean", "Std Dev", "Min", "Max"]

# Model
N_LAYERS        = 2
WEIGHT_INIT_STD = 0.01

# Training
EPOCHS        = int(os.getenv("EPOCHS", "50"))
BATCH_SIZE    = 32
LEARNING_RATE = 0.00005
LR_STEP_SIZE  = 10
LR_GAMMA      = 0.95
GRAD_CLIP     = 1.0
WEIGHT_INIT_STD = 0.01

# Evaluation
EVAL_EVERY   = 10
EVAL_SAMPLES = 200

# Experiment: run with 2, 3, then 4 features to show scaling
FEATURE_SWEEP = [2, 3, 4]

# Output
FIGURES_DIR = "figures"