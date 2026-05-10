# scripts/collect_infested.py
"""
Infested-only collection protocol.

Run 1 : calibration (manual segment confirmation per rotation).
Runs 2-N : automatic scans of confirmed infested segments only.

Run from the project root:
    python scripts/collect_infested.py
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
    OUTPUT_FILE, ROTATE_STEPS, TOTAL_RUNS,
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


def save_row(eggplant_id, readings):
    """Always saves as Infested — this script only collects infested samples."""
    with open(OUTPUT_FILE, 'a', newline='') as f:
        csv.writer(f).writerow(
            ['Infested', eggplant_id, time.strftime("%Y-%m-%d %H:%M:%S")] + readings
        )


# ── Display readings ──────────────────────────────────────────────────────────

def print_readings(r0, r1, seg_num):
    bands = ["UV/Blue   Ch1-6 ", "Grn/Ylw  Ch7-12", "Red/NIR  Ch13-18"]
    print(f"\n  {'─'*60}")
    print(f"  Segment {seg_num} readings (uW/cm²)")
    print(f"  {'Band':<18} {'Sensor 0':^25} {'Sensor 1':^25}")
    print(f"  {'─'*60}")
    for i, lbl in enumerate(bands):
        c0 = (r0 or [0] * 18)[i * 6:(i + 1) * 6]
        c1 = (r1 or [0] * 18)[i * 6:(i + 1) * 6]
        print(
            f"  {lbl}  "
            f"{'  '.join(f'{v:7.1f}' for v in c0)}  "
            f"{'  '.join(f'{v:7.1f}' for v in c1)}"
        )
    print(f"  {'─'*60}")


# ── Calibration run ───────────────────────────────────────────────────────────

def calibration_run(
    s0: SpectralSensor,
    s1: SpectralSensor,
    motor: Motor,
    eid: str,
) -> set:
    """Scan all 3 segments; user confirms which are infested."""
    print("\n  ── CALIBRATION RUN ──────────────────────────────")
    infested = set()
    saved    = 0

    for step in range(3):
        print(f"\n  [Segment {step + 1}/3] Scanning...")
        r0 = s0.read()
        r1 = s1.read()
        if r0 or r1:
            print_readings(r0, r1, step + 1)

        while True:
            ans = input(f"\n  Segment {step + 1} INFESTED? (Y/N): ").strip().upper()
            if ans in ('Y', 'N'):
                break
            print("  Enter Y or N.")

        if ans == 'Y':
            infested.add(step)
            if r0: save_row(eid + '_S0', r0); saved += 1
            if r1: save_row(eid + '_S1', r1); saved += 1
            print("  ✓ Saved as Infested.")
        else:
            print("  ✗ Discarded.")

        if step < 2:
            motor.move_forward(COLLECT_MOTOR_SPEED, COLLECT_MOVE_DURATION)

    motor.return_home(COLLECT_MOTOR_SPEED, COLLECT_RETURN_DURATION)

    if infested:
        print(f"\n  Infested segments: {sorted(s + 1 for s in infested)}")
        print(
            f"  Saved {saved} readings. "
            f"Auto-scanning these for {TOTAL_RUNS - 1} more runs."
        )
    else:
        print("\n  No infested segments found.")

    return infested


# ── Auto runs ─────────────────────────────────────────────────────────────────

def auto_run(
    s0: SpectralSensor,
    s1: SpectralSensor,
    motor: Motor,
    eid: str,
    infested_segs: set,
) -> int:
    """Scan only the confirmed infested segments; return number of rows saved."""
    if not infested_segs:
        return 0

    saved = 0
    for step in range(3):
        if step in infested_segs:
            r0 = s0.read()
            r1 = s1.read()
            print_readings(r0, r1, step + 1)
            if r0: save_row(eid + '_S0', r0); saved += 1
            if r1: save_row(eid + '_S1', r1); saved += 1

        if step < 2:
            motor.move_forward(COLLECT_MOTOR_SPEED, COLLECT_MOVE_DURATION)

    motor.return_home(COLLECT_MOTOR_SPEED, COLLECT_RETURN_DURATION)
    return saved


# ── Per-rotation session ──────────────────────────────────────────────────────

def collect_rotation(
    s0: SpectralSensor,
    s1: SpectralSensor,
    motor: Motor,
    eid: str,
    rot_num: int,
) -> int:
    print(f"\n{'=' * 60}")
    print(f"  ROTATION {rot_num}/{ROTATE_STEPS}  —  {eid}")
    print(f"{'=' * 60}")

    if rot_num > 1:
        input(f"\n  Rotate ~{360 // ROTATE_STEPS}° then press [ENTER]: ")
    input("  Close box and press [ENTER] for calibration scan: ")

    inf_segs = calibration_run(s0, s1, motor, eid)
    if not inf_segs:
        print("  Skipping remaining runs for this rotation.")
        return 0

    total = 0
    for run in range(2, TOTAL_RUNS + 1):
        print(f"\n  Auto-run {run}/{TOTAL_RUNS}...", end=' ')
        n = auto_run(s0, s1, motor, eid, inf_segs)
        total += n
        print(f"({n} saved)")

    return total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== INFESTED-ONLY COLLECTION ===")
    print(f"Protocol: 1 calibration + {TOTAL_RUNS - 1} auto runs per rotation\n")

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

    # Short initial home (1.5 s) — just clears the belt without a full return sweep
    print("Homing...")
    motor.return_home(COLLECT_MOTOR_SPEED, 1.5)

    print("\nWarm-up (2 min). Place eggplant and close box.")
    time.sleep(120)
    print("Warm-up done.\n")

    ensure_csv()

    try:
        while True:
            print("\n" + "=" * 60)
            eid = input("Eggplant ID (e.g. I11) or Q to quit: ").strip().upper()
            if eid == 'Q':
                break
            if not eid:
                print("  ID cannot be empty.")
                continue

            total = 0
            for rot in range(1, ROTATE_STEPS + 1):
                total += collect_rotation(sensor0, sensor1, motor, eid, rot)
                print(f"\n  Rotation {rot} done. Total saved so far: {total}")
            print(f"\n  [{eid}] complete. Total saved: {total}")

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
