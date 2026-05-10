# Setup Guide

## Requirements

- Raspberry Pi 4 (recommended: 4GB RAM)
- Python 3.10+
- All hardware components connected and wired (see Hardware section in README)

---

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/talong-ai.git
cd talong-ai
```

---

## 2. Create a Virtual Environment

Using `micromamba` (recommended on Pi):

```bash
micromamba create -n talong_stable python=3.10
micromamba activate talong_stable
```

Or using standard `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install tensorflow
pip install scikit-learn scipy numpy pandas
pip install pygame
pip install smbus2
pip install evdev
pip install sparkfun-qwiic-as7265x
pip install sparkfun-qwiic-scmd
```

> **Note:** TensorFlow on Raspberry Pi may require a specific wheel depending on
> your OS and architecture. If the standard install fails, try:
> ```bash
> pip install tensorflow-aarch64
> ```

---

## 4. Hardware Wiring

| Component | Interface | Address / Port |
|---|---|---|
| TCA9548A Multiplexer | I²C | 0x70 |
| AS7265x Sensor 0 | I²C via MUX | Port 0 |
| AS7265x Sensor 1 | I²C via MUX | Port 3 |
| SCMD Motor Driver | I²C | default |
| Macro pad | USB | `/dev/input/by-id/...` |

Enable I²C on the Pi if not already done:

```bash
sudo raspi-config
# Interface Options → I2C → Enable
```

Verify devices are detected:

```bash
i2cdetect -y 1
```

---

## 5. Configuration

All hardware, model, and data collection settings are in `config.py`. Review and
adjust if your wiring or paths differ from the defaults:

```python
I2C_BUS       = 1
MUX_ADDR      = 0x70
SENSOR_0_PORT = 0
SENSOR_1_PORT = 3
MOTOR_SPEED   = 65
```

---

## 6. Train a Model

First, collect data using the collection scripts (see [Data Collection](#7-data-collection)),
then train:

```bash
# CNN (default)
python scripts/train.py

# Random Forest (alternative)
python scripts/train_rf.py
```

Trained models are saved to `models/`. Update `MODEL_NAME` or `RF_MODEL_NAME`
in `config.py` to point to your output file. To switch backends, set `USE_RF`
in `config.py` to `True` or `False`.

---

## 7. Data Collection

**General collection (healthy + infested):**

```bash
python scripts/collect.py
```

**Infested-only collection (calibration + auto-scan protocol):**

```bash
python scripts/collect_infested.py
```

Both scripts will prompt you for an eggplant ID and walk you through the
scanning process. Output is appended to the CSV file defined by `OUTPUT_FILE`
in `config.py`.

---

## 8. Evaluate the Model

Run a 5-fold leave-one-eggplant-out evaluation:

```bash
python scripts/evaluate_cnn.py
```

Run a quick SVM vs RF baseline comparison:

```bash
python scripts/baseline.py
```

---

## 9. Run the System

```bash
python main.py
```

This spawns the Pygame display process and starts the hardware loop. Use
the macro pad or keyboard to control the system:

| Key | Action |
|---|---|
| `1` / `KEY_A` | Power toggle (on → off → shutdown) |
| `2` / `KEY_B` | Trigger scan |
| `3` / `KEY_C` | Cancel scan, return home |
| `4` / `KEY_D` | Reset (restart main.py) |
| `Enter` | Trigger scan (keyboard) |
| `c` + Enter | Cancel scan (keyboard) |
| `q` + Enter | Quit |

---

## 10. Auto-start on Boot (Optional)

To have the system launch automatically on boot using tmux:

```bash
sudo apt install tmux
```

Add to your `~/.bashrc` or a systemd service:

```bash
tmux new-session -d -s talongs 'python3 /path/to/talong-ai/main.py'
```

Then run the macro listener in a separate session:

```bash
tmux new-session -d -s talongs_keys 'python3 /path/to/talong-ai/macro_listener.py'
```

---

## Troubleshooting

**Sensors not detected**
- Run `i2cdetect -y 1` to confirm the MUX is at `0x70`
- Check that the correct MUX port numbers match `config.py`

**TensorFlow import errors**
- Confirm you're in the correct virtual environment
- Try `pip install tensorflow-aarch64` for ARM builds

**Pygame display fails**
- Ensure `DISPLAY=:0` is set: `export DISPLAY=:0`
- If running headless, check that a display server is running

**Macro pad not found**
- Check `ls /dev/input/by-id/` for the correct device path
- Update the path in `macro_listener.py` if it differs from the default
