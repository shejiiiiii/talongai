# config.py
# Single source of truth for the entire project.
# Previously split across config.py and data_collection/config_collect.py.

# ── Inference backend ──────────────────────────────────────────────────────────
# Set USE_RF = True to switch to Random Forest; False uses the SE-CNN.
# This flag is read by src/analyzer.py — no more commenting/uncommenting blocks.
USE_RF = False

# ── Dataset / Models ───────────────────────────────────────────────────────────
DATA_FILE     = "data/eggplant_spectral_data_v5.csv"   # training dataset (was v3 — fixed)
OUTPUT_FILE   = "data/eggplant_spectral_data_v6.csv"   # data collection output

MODEL_DIR     = "models"
MODEL_NAME    = f"{MODEL_DIR}/eggplant_pest_model_v5.keras"
RF_MODEL_NAME = f"{MODEL_DIR}/eggplant_pest_rf_v4.pkl"
CNN_CLASSES   = f"{MODEL_DIR}/classes.npy"
RF_CLASSES    = f"{MODEL_DIR}/rf_classes.npy"

# ── Hardware / I2C ─────────────────────────────────────────────────────────────
INPUT_SHAPE   = (18, 1)   # 18 spectral channels → (timesteps, features) for CNN
I2C_BUS       = 1
MUX_ADDR      = 0x70

SENSOR_0_PORT = 0
SENSOR_1_PORT = 3

# ── Motor — inference (production scanning) ────────────────────────────────────
# Slower speed and shorter duration for controlled single-pass inference.
MOTOR_SPEED     = 65
MOVE_DURATION   = 0.6
RETURN_DURATION = 4.3

# ── Motor — data collection ────────────────────────────────────────────────────
# Higher speed / longer duration used during the collection protocol.
COLLECT_MOTOR_SPEED     = 100
COLLECT_MOVE_DURATION   = 1.5
COLLECT_RETURN_DURATION = 4.0

# ── CNN Training ───────────────────────────────────────────────────────────────
TEST_SIZE     = 0.2
RANDOM_STATE  = 42
BATCH_SIZE    = 32
EPOCHS        = 200
LEARNING_RATE = 0.001
Z_THRESHOLD   = 2.5   # std-dev cutoff for per-eggplant outlier removal

# ── Inference ──────────────────────────────────────────────────────────────────
# Segments whose confidence falls below CONFIDENCE_THRESHOLD are flipped.
CONFIDENCE_THRESHOLD = 60
NUM_SEGMENTS         = 5

# ── Data collection protocol ───────────────────────────────────────────────────
PASSES_PER_SIDE_HEALTHY  = 12
PASSES_PER_SIDE_INFESTED = 20
ROTATE_STEPS             = 4    # rotation positions per eggplant (~45° each)
TOTAL_RUNS               = 20   # collect_infested: 1 calibration + 19 auto runs

# ── Evaluation ─────────────────────────────────────────────────────────────────
EVAL_FOLDS = 5   # k for leave-one-eggplant-out cross-validation
