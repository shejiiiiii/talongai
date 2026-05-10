# scripts/collect.py
"""
Data collection with per-segment labeling for infested eggplants.

Run from the project root:
    python scripts/collect.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import time

import smbus2

from config import (
    I2C_BUS, MUX_ADDR,
    SENSOR_0_PORT, SENSOR_1_PORT,
    COLLECT_MOTOR_SPEED, COLLECT_MOVE_DURATION, COLLECT_RETURN_DURATION,
    OUTPUT_FILE,
    PASSES_PER_SIDE_HEALTHY, PASSES_PER_SIDE_INFESTED, ROTATE_STEPS,
)
from src.hardware import Multiplexer, SpectralSensor, Motor


# ── CSV helpers ───────────────────────────────────────────────────────────────

def ensure_csv():
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['Label', 'Eggplant_ID', 'Timestamp'] +
                [f'Ch_{i}' for i in range(1, 19)]
            )
        print(f"  Created: {OUTPUT_FILE}")
    else:
        print(f"  Appending to: {OUTPUT_FILE}")


def save_row(label, eggplant_id, readings):
    with open(OUTPUT_FILE, 'a', newline='') as f:
        csv.writer(f).writerow(
            [label, eggplant_id, time.strftime("%Y-%m-%d %H:%M:%S")] + readings
        )


# ── Infested segment prompt ───────────────────────────────────────────────────

def ask_infested_segments() -> set:
    """
    Ask which of the 3 conveyor segments face the infested area.
    Returns a set of 0-indexed segment numbers, e.g. {0, 1}.
    'all' → {0, 1, 2},  'none' → {}
    """
    while True:
        raw = input(
            "  Which segments face the INFESTED area? "
            "(1/2/3 or combinations, 'all', 'none'): "
        ).strip().lower()
        if raw == 'none': return set()
        if raw == 'all':  return {0, 1, 2}
        try:
            parts = [int(x) - 1 for x in raw.replace(',', ' ').split()]
            if all(0 <= p <= 2 for p in parts):
                return set(parts)
        except ValueError:
            pass
        print("  Enter segment numbers (e.g. 1 2), 'all', or 'none'.")


# ── Conveyor pass ─────────────────────────────────────────────────────────────

def run_pass(
    sensor0: SpectralSensor,
    sensor1: SpectralSensor,
    motor: Motor,
    eggplant_id: str,
    label: str,
    infested_segments: set = None,
) -> dict:
    """
    One full conveyor pass (3 segments).

    Healthy  : all segments saved as 'Healthy'.
    Infested : only segments in infested_segments saved; others discarded.
    """
    saved = {'Healthy': 0, 'Infested': 0, 'Discarded': 0}

    for step in range(3):
        r0 = sensor0.read()
        r1 = sensor1.read()

        if label == 'Healthy':
            keep, seg_label = True, 'Healthy'
        elif infested_segments is not None and step in infested_segments:
            keep, seg_label = True, 'Infested'
        else:
            keep, seg_label = False, None

        if keep:
            if r0: save_row(seg_label, eggplant_id + '_S0', r0); saved[seg_label] += 1
            if r1: save_row(seg_label, eggplant_id + '_S1', r1); saved[seg_label] += 1
        else:
            saved['Discarded'] += 1

        if step < 2:
            motor.move_forward(COLLECT_MOTOR_SPEED, COLLECT_MOVE_DURATION)

    motor.return_home(COLLECT_MOTOR_SPEED, COLLECT_RETURN_DURATION)
    return saved


# ── Per-eggplant collection ───────────────────────────────────────────────────

def collect_eggplant(
    sensor0: SpectralSensor,
    sensor1: SpectralSensor,
    motor: Motor,
    label: str,
    eggplant_id: str,
):
    total = {'Healthy': 0, 'Infested': 0, 'Discarded': 0}

    for rot in range(ROTATE_STEPS):
        print(f"\n  Rotation {rot + 1}/{ROTATE_STEPS}")
        if rot > 0:
            input("  Rotate eggplant ~45° then press [ENTER]: ")

        inf_segs = None
        if label == 'Infested':
            print("  Segment layout: [1]=front  [2]=middle  [3]=back")
            inf_segs = ask_infested_segments()
            if not inf_segs:
                print("  No infested segments — skipping rotation.")
                # Advance belt through all segments then return home
                motor.move_forward(COLLECT_MOTOR_SPEED, COLLECT_MOVE_DURATION)
                motor.move_forward(COLLECT_MOTOR_SPEED, COLLECT_MOVE_DURATION)
                motor.return_home(COLLECT_MOTOR_SPEED, COLLECT_RETURN_DURATION)
                continue

        passes = PASSES_PER_SIDE_INFESTED if label == 'Infested' else PASSES_PER_SIDE_HEALTHY
        for p in range(passes):
            print(f"    Pass {p + 1}/{passes}...", end=' ')
            saved = run_pass(sensor0, sensor1, motor, eggplant_id, label, inf_segs)
            for k in total:
                total[k] += saved.get(k, 0)
            print(
                f"I:{saved.get('Infested', 0)}  "
                f"H:{saved.get('Healthy', 0)}  "
                f"discarded:{saved.get('Discarded', 0)}"
            )

    print(
        f"\n  [{eggplant_id}] done — "
        f"I:{total['Infested']}  H:{total['Healthy']}  "
        f"discarded:{total['Discarded']}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== EGGPLANT DATA COLLECTION ===\n")

    bus     = smbus2.SMBus(I2C_BUS)
    mux     = Multiplexer(bus, MUX_ADDR)
    sensor0 = SpectralSensor(mux, SENSOR_0_PORT)
    sensor1 = SpectralSensor(mux, SENSOR_1_PORT)

    if not sensor0.initialize() or not sensor1.initialize():
        print("Sensor failure. Exiting.")
        bus.close()
        return

    motor = Motor()
    if not motor.initialize():
        print("Motor driver not found. Exiting.")
        bus.close()
        return

    print("Homing conveyor...")
    motor.return_home(COLLECT_MOTOR_SPEED, COLLECT_RETURN_DURATION)

    print("\nSensor warm-up (2 min). Place eggplant and close box.")
    time.sleep(120)
    print("Warm-up done.\n")

    ensure_csv()

    try:
        while True:
            print("\n" + "=" * 50)
            choice = input("Label  H=Healthy / I=Infested / Q=quit: ").strip().upper()
            if choice == 'Q':
                break
            if choice not in ('H', 'I'):
                print("  Enter H, I, or Q.")
                continue

            label       = 'Healthy' if choice == 'H' else 'Infested'
            eggplant_id = input("Eggplant ID (e.g. H01, I03): ").strip().upper()
            if not eggplant_id:
                print("  ID cannot be empty.")
                continue

            input("  Place eggplant, close box, press [ENTER]: ")
            collect_eggplant(sensor0, sensor1, motor, label, eggplant_id)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        try:
            motor.disable()
            sensor0.disable_light()
            sensor1.disable_light()
            mux.select('off')
            bus.close()
        except Exception as e:
            print(f"Cleanup error: {e}")
        print("Exited.")


if __name__ == '__main__':
    main()
