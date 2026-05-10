# TalongAI — Eggplant Pest Detection System

A Raspberry Pi embedded system that uses near-infrared (NIR) spectroscopy to 
detect fruit borer infestation in eggplants (*talong* in Filipino) — non-destructively 
and in real time.

---

## Overview

Two AS7265x 18-channel spectral sensors are mounted above a motorized conveyor belt. 
As an eggplant passes beneath them in segments, each sensor captures a full NIR 
spectral reading. A trained machine learning model (SE-CNN or Random Forest) then 
classifies each segment as **Healthy** or **Infested**, with majority voting used 
to produce a final per-eggplant result. Results are displayed on a 1024×600 
touchscreen dashboard built with Pygame.

---

## Features

- **Dual spectral sensors** — two AS7265x sensors (18 channels, 410–940 nm) 
  positioned side-by-side for simultaneous dual-eggplant scanning
- **Motorized conveyor** — SCMD-controlled DC belt with automatic homing
- **SE-CNN model** — Squeeze-and-Excitation 1D CNN trained on 7,000+ labeled 
  spectral scans with Savitzky-Golay smoothing and SNV normalization
- **Random Forest alternative** — switchable RF backend with pairwise channel 
  ratio features for interpretability
- **Confidence thresholding** — low-confidence segment predictions are flipped 
  to reduce false positives
- **Majority voting** — any infested segment flags the whole eggplant
- **Pygame dashboard** — animated boot screen, live scan status, per-eggplant 
  result panels, macro key guide
- **Macro pad control** — 4-button USB macro keyboard for power, scan, cancel, 
  and reset without a touchscreen
- **Spawned display process** — TensorFlow and Pygame run in separate processes 
  to avoid GPU/GL conflicts on ARM

---

## Hardware

| Component | Details |
|---|---|
| SBC | Raspberry Pi 4 |
| Spectral sensors | SparkFun AS7265x Triad (×2) |
| I²C multiplexer | TCA9548A (0x70) |
| Motor driver | SparkFun SCMD |
| Display | 1024×600 HDMI touchscreen |
| Input | 4-key USB macro pad |

---

## Project Structure
├── main.py                  # Entry point; spawns display process, runs hardware loop
├── config.py                # Shared constants (model paths, hardware, training)
├── analyzer.py              # PestAnalyzer — CNN or RF inference + majority vote
├── model.py                 # SE-CNN architecture (TensorFlow/Keras)
├── dataset.py               # Data loading, SNV normalization, augmentation
├── hardware.py              # Multiplexer, SpectralSensor, Motor drivers
├── display.py               # Pygame dashboard (boot screen + live dashboard)
├── display_process.py       # Display subprocess entry point (TF-free)
├── macro_listener.py        # evdev macro pad listener + power state management
├── blank_screen.py          # Standby black screen (device off state)
├── train.py                 # CNN training script
├── train_rf.py              # Random Forest training script
├── baseline.py              # SVM + RF baseline comparison
├── evaluate_cnn.py          # 5-fold leave-one-eggplant-out CNN evaluation
├── collect.py               # Interactive data collection (healthy + infested)
├── collect_infested.py      # Infested-only collection with calibration protocol
├── config_collect.py        # Data collection config
└── eggplant_spectral_data_v5.csv  # Labeled spectral dataset (7,276 scans)
---

## Switching the Inference Backend

The system supports both CNN and Random Forest. To switch, follow the comments 
in `analyzer.py` — comment out the CNN block and uncomment the RF block, then 
point `RF_MODEL_NAME` in `config.py` to your trained `.pkl` file.

---

## Data Collection

Spectral data was collected using `collect.py` (general) and `collect_infested.py` 
(a calibration-then-auto-scan protocol for infested specimens). Each eggplant is 
scanned across 4 rotation positions in 3 conveyor segments. Rows are labeled at 
the segment level — only segments facing confirmed infestation sites are saved as 
`Infested`.

---

## Model Training

```bash
# CNN
python train.py

# Random Forest
python train_rf.py

# Baseline comparison (SVM vs RF)
python baseline.py

# 5-fold leave-one-eggplant-out evaluation
python evaluate_cnn.py
```

---

## Dependencies

- Python 3.10+
- TensorFlow 2.x
- scikit-learn
- scipy, numpy, pandas
- pygame
- smbus2
- qwiic_as7265x, qwiic_scmd
- evdev (macro pad listener)

---

## License

MIT

