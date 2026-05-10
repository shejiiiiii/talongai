# Code Style Guide

This document describes the conventions used throughout the TalongAI codebase.
Follow these when contributing or extending the project.

---

## General

- **Language:** Python 3.10+
- **Line length:** 88 characters max (Black-compatible)
- **Indentation:** 4 spaces — no tabs
- **Encoding:** UTF-8 everywhere

---

## Naming Conventions

| Thing | Convention | Example |
|---|---|---|
| Variables | `snake_case` | `motor_speed`, `scan_result` |
| Functions | `snake_case` | `get_readings()`, `run_pass()` |
| Classes | `PascalCase` | `SpectralSensor`, `PestAnalyzer` |
| Constants | `UPPER_SNAKE_CASE` | `MUX_ADDR`, `MOTOR_SPEED` |
| Private methods | leading underscore | `_preprocess()`, `_load_model()` |
| Files/modules | `snake_case` | `hardware.py`, `display_process.py` |

---

## File Structure

Each file should open with a module-level docstring explaining its role,
followed by imports, then constants, then classes/functions.

```python
# src/example.py
"""
One-line summary of what this module does.

Longer explanation if needed — architecture decisions, caveats, etc.
"""

import os                        # stdlib first
import numpy as np               # third-party second
from config import MOTOR_SPEED   # local last
```

Imports are ordered: **stdlib → third-party → local**, with a blank line
between each group.

---

## Comments and Docstrings

- Use `#` comments to explain *why*, not *what*
- Section dividers use the established style already in the codebase:

```python
# ── Section Name ──────────────────────────────────────────────────────────────
```

- Public classes and functions get a short docstring:

```python
def predict(self, readings_list: list) -> tuple[str, float, bool]:
    """
    Returns (label, confidence_pct, was_flipped).

    Segments whose confidence < CONFIDENCE_THRESHOLD are flipped
    to the opposite label.
    """
```

- One-liners are fine for simple private helpers:

```python
def stop(self):
    """Immediately cut drive to both motors."""
```

- Inline comments should be separated by two spaces and kept brief:

```python
x = MaxPooling1D(pool_size=2)(x)   # 18 → 9
```

---

## Type Hints

Use type hints on all public function signatures:

```python
def initialize(self) -> bool: ...
def augment_data(X: np.ndarray, y: np.ndarray, copies: int = 2) -> tuple[np.ndarray, np.ndarray]: ...
```

Local variables and private helpers don't need hints unless the type is
non-obvious.

---

## Configuration

- **All hardware constants, paths, and tunable parameters belong in `config.py`
  or `config_collect.py`** — not hardcoded in scripts
- Scripts import from config; they never define their own magic numbers

```python
# Good
from config import MOTOR_SPEED, MOVE_DURATION

# Bad
motor.set_drive(0, 0, 65)   # where did 65 come from?
```

---

## Hardware / I2C Code

- Always guard hardware init with a retry loop and a clear failure message
- Always disable bulbs and motors in `finally` blocks — hardware should
  never be left in a live state if the script crashes
- Use `time.sleep()` generously around I²C operations — sensors need time
  to settle

```python
# Good
try:
    sensor.enable_bulb(0)
    sensor.take_measurements()
finally:
    sensor.disable_bulb(0)
```

---

## Machine Learning Code

- Preprocessing steps (Savitzky-Golay, SNV) must be applied consistently
  in both training (`dataset.py`) and inference (`analyzer.py`)
- Test sets are never augmented — only the training split
- Class weights are always computed from the training split, not the full dataset
- Label encoding classes are saved to `.npy` alongside the model so inference
  can verify the class order

---

## Switching Backends (CNN ↔ RF)

The codebase supports two inference backends. Switching is done by
commenting/uncommenting clearly marked blocks in `analyzer.py`. Keep both
blocks intact and clearly labeled — do not delete the inactive backend.

```python
# ── CNN ────────────────────────────────────────────────────────────────
# model = keras_load_model(MODEL_NAME)

# ── RF (uncomment to switch) ───────────────────────────────────────────
model = pickle.load(...)
```

---

## Multiprocessing

- TensorFlow and Pygame must never share a process on ARM — always use
  `multiprocessing.set_start_method("spawn")` and keep TF imports out of
  the display process entirely
- IPC between the main process and display process uses a `multiprocessing.Queue`
  of state dicts — keep messages small and non-blocking (`put_nowait`)
- The display process must never import anything from `src/` that transitively
  imports TensorFlow

---

## Error Handling

- Hardware errors should print a warning and degrade gracefully — the system
  should still run (in a reduced capacity) if one sensor or the motor fails
- Use specific `except` clauses — avoid bare `except:` outside of cleanup
  `finally` blocks
- Cleanup code (motor stop, mux off, bus close) always goes in `finally`

```python
# Good
try:
    motor.disable()
except Exception as e:
    print(f"Cleanup error: {e}")   # log it, don't crash

# Bad
except:
    pass   # silent failure, no trace
```

---

## What Not to Do

- Don't import `tensorflow` in any file that the display process might load
- Don't hardcode sensor ports, addresses, or model paths outside of `config.py`
- Don't commit trained model files (`.keras`, `.pkl`) — they are in `.gitignore`
- Don't commit `state.json` — it is device-specific runtime state
- Don't augment the test set
