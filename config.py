# config.py

# ── Dataset / Models ───────────────────────────────────────────────────────────
DATA_FILE     = "eggplant_spectral_data_v3.csv"
MODEL_NAME    = "data_collection/eggplant_pest_model_v5.keras"
RF_MODEL_NAME = "data_collection/eggplant_pest_rf_v2.pkl"   # used when switching to RF

# ── Hardware / I2C ─────────────────────────────────────────────────────────────
INPUT_SHAPE = (18, 1)   # 18 spectral channels
I2C_BUS     = 1
MUX_ADDR    = 0x70

# Mux physical port numbers
SENSOR_0_PORT = 0
SENSOR_1_PORT = 3

# ── Motor ──────────────────────────────────────────────────────────────────────
MOTOR_SPEED     = 65
MOVE_DURATION   = 0.6
RETURN_DURATION = 4.3

# ── CNN Training ───────────────────────────────────────────────────────────────
TEST_SIZE     = 0.2
RANDOM_STATE  = 42
BATCH_SIZE    = 32
EPOCHS        = 200
LEARNING_RATE = 0.001

# ── Inference ──────────────────────────────────────────────────────────────────
# Segments whose confidence falls below this % are flipped to the opposite label
CONFIDENCE_THRESHOLD = 60
NUM_SEGMENTS         = 5
