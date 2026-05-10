# data_collection/config_collect.py

# Hardware (matches main config)
I2C_BUS       = 1
MUX_ADDR      = 0x70
SENSOR_0_PORT = 0
SENSOR_1_PORT = 3

# Motor — collection uses slightly different speed/timing
MOTOR_SPEED     = 100
MOVE_DURATION   = 1.5
RETURN_DURATION = 4.0

# Collection settings
OUTPUT_FILE              = "eggplant_spectral_data_v6.csv"
PASSES_PER_SIDE_HEALTHY  = 12
PASSES_PER_SIDE_INFESTED = 20
ROTATE_STEPS             = 4    # rotation positions per eggplant (~45° each)
