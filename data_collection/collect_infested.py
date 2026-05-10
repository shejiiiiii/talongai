# data_collection/collect_infested.py
"""
Infested-only collection protocol.
Run 1: calibration (manual segment confirmation).
Runs 2-N: automatic scans of confirmed segments only.
"""

import sys, os, csv, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import smbus2
import qwiic_as7265x
import qwiic_scmd

from data_collection.config_collect import (
    I2C_BUS, MUX_ADDR,
    SENSOR_0_PORT, SENSOR_1_PORT,
    MOTOR_SPEED, MOVE_DURATION, RETURN_DURATION,
    OUTPUT_FILE,
)

bus = smbus2.SMBus(I2C_BUS)

TOTAL_RUNS   = 20   # 1 calibration + 19 automatic
ROTATE_STEPS = 4


# ── Hardware ──────────────────────────────────────────────────────────────────

def select_mux(ch):
    bus.write_byte(MUX_ADDR, 0x00 if ch == 'off' else 1 << ch)

def init_sensor(channel):
    select_mux(channel); time.sleep(0.5)
    sensor = qwiic_as7265x.QwiicAS7265x()
    for _ in range(3):
        try:
            if sensor.begin():
                sensor.soft_reset(); time.sleep(1)
                sensor.set_gain(3)
                sensor.set_integration_cycles(50)
                sensor.disable_indicator()
                sensor.set_bulb_current(12.5, 0)
                sensor.disable_bulb(0)
                print(f"  Sensor ch{channel} ready.")
                return sensor
        except OSError:
            time.sleep(0.5)
    print(f"  ERROR: Sensor ch{channel} not found.")
    return None

def get_readings(sensor, channel):
    for _ in range(3):
        try:
            select_mux(channel); time.sleep(0.1)
            sensor.enable_bulb(0); time.sleep(0.1)
            sensor.take_measurements()
            r = [
                sensor.get_calibrated_a(), sensor.get_calibrated_b(),
                sensor.get_calibrated_c(), sensor.get_calibrated_d(),
                sensor.get_calibrated_e(), sensor.get_calibrated_f(),
                sensor.get_calibrated_g(), sensor.get_calibrated_h(),
                sensor.get_calibrated_i(), sensor.get_calibrated_j(),
                sensor.get_calibrated_k(), sensor.get_calibrated_l(),
                sensor.get_calibrated_r(), sensor.get_calibrated_s(),
                sensor.get_calibrated_t(), sensor.get_calibrated_u(),
                sensor.get_calibrated_v(), sensor.get_calibrated_w(),
            ]
            sensor.disable_bulb(0)
            return r
        except OSError:
            try: sensor.disable_bulb(0)
            except: pass
            time.sleep(0.2)
    print("  [Warning] Sensor busy, skipping.")
    return None

def motor_forward(motor):
    motor.set_drive(0, 0, MOTOR_SPEED); motor.set_drive(1, 1, MOTOR_SPEED)
    time.sleep(MOVE_DURATION)
    motor.set_drive(0, 0, 0); motor.set_drive(1, 1, 0)
    time.sleep(0.2)

def motor_home(motor, dur):
    motor.set_drive(0, 1, MOTOR_SPEED); motor.set_drive(1, 0, MOTOR_SPEED)
    time.sleep(dur)
    motor.set_drive(0, 0, MOTOR_SPEED); motor.set_drive(1, 1, MOTOR_SPEED)
    time.sleep(0.5)
    motor.set_drive(0, 0, 0); motor.set_drive(1, 0, 0)


# ── CSV ───────────────────────────────────────────────────────────────────────

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
        c0 = (r0 or [0]*18)[i*6:(i+1)*6]
        c1 = (r1 or [0]*18)[i*6:(i+1)*6]
        print(f"  {lbl}  "
              f"{'  '.join(f'{v:7.1f}' for v in c0)}  "
              f"{'  '.join(f'{v:7.1f}' for v in c1)}")
    print(f"  {'─'*60}")


# ── Calibration run ───────────────────────────────────────────────────────────

def calibration_run(s0, s1, motor, eid) -> set:
    """Scan all 3 segments; user confirms which are infested."""
    print("\n  ── CALIBRATION RUN ──────────────────────────────")
    infested = set()
    saved    = 0

    for step in range(3):
        print(f"\n  [Segment {step+1}/3] Scanning...")
        r0 = get_readings(s0, SENSOR_0_PORT)
        r1 = get_readings(s1, SENSOR_1_PORT)
        if r0 or r1:
            print_readings(r0, r1, step+1)

        while True:
            ans = input(f"\n  Segment {step+1} INFESTED? (Y/N): ").strip().upper()
            if ans in ('Y','N'): break
            print("  Enter Y or N.")

        if ans == 'Y':
            infested.add(step)
            if r0: save_row(eid + '_S0', r0); saved += 1
            if r1: save_row(eid + '_S1', r1); saved += 1
            print(f"  ✓ Saved as Infested.")
        else:
            print(f"  ✗ Discarded.")

        if step < 2:
            motor_forward(motor)

    motor_home(motor, RETURN_DURATION)

    if infested:
        print(f"\n  Infested segments: {sorted(s+1 for s in infested)}")
        print(f"  Saved {saved} readings. "
              f"Auto-scanning these for {TOTAL_RUNS-1} more runs.")
    else:
        print("\n  No infested segments found.")
    return infested


# ── Auto runs ─────────────────────────────────────────────────────────────────

def auto_run(s0, s1, motor, eid, infested_segs, run_num) -> int:
    if not infested_segs:
        return 0
    saved = 0
    for step in range(3):
        if step in infested_segs:
            r0 = get_readings(s0, SENSOR_0_PORT)
            r1 = get_readings(s1, SENSOR_1_PORT)
            print_readings(r0, r1, step+1)
            if r0: save_row(eid + '_S0', r0); saved += 1
            if r1: save_row(eid + '_S1', r1); saved += 1
        if step < 2:
            motor_forward(motor)
    motor_home(motor, RETURN_DURATION)
    return saved


# ── Per-rotation session ──────────────────────────────────────────────────────

def collect_rotation(s0, s1, motor, eid, rot_num) -> int:
    print(f"\n{'='*60}")
    print(f"  ROTATION {rot_num}/{ROTATE_STEPS}  —  {eid}")
    print(f"{'='*60}")
    if rot_num > 1:
        input(f"\n  Rotate ~{360//ROTATE_STEPS}° then press [ENTER]: ")
    input("  Close box and press [ENTER] for calibration scan: ")

    inf_segs = calibration_run(s0, s1, motor, eid)
    if not inf_segs:
        print("  Skipping remaining runs for this rotation.")
        return 0

    total = 0
    for run in range(2, TOTAL_RUNS + 1):
        print(f"\n  Auto-run {run}/{TOTAL_RUNS}...", end=' ')
        n = auto_run(s0, s1, motor, eid, inf_segs, run)
        total += n
        print(f"({n} saved)")
    return total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== INFESTED-ONLY COLLECTION ===")
    print(f"Protocol: 1 calibration + {TOTAL_RUNS-1} auto runs per rotation\n")

    s0 = init_sensor(SENSOR_0_PORT)
    s1 = init_sensor(SENSOR_1_PORT)
    if not s0 or not s1:
        print("Sensor failure. Exiting."); return

    motor = qwiic_scmd.QwiicScmd()
    if not motor.connected:
        print("Motor driver not found. Exiting."); return
    motor.begin(); motor.enable()

    print("Homing..."); motor_home(motor, 1.5)
    print("\nWarm-up (2 min). Place eggplant and close box.")
    time.sleep(120)
    print("Warm-up done.\n")

    ensure_csv()

    try:
        while True:
            print("\n" + "="*60)
            eid = input("Eggplant ID (e.g. I11) or Q to quit: ").strip().upper()
            if eid == 'Q': break
            if not eid: print("  ID cannot be empty."); continue

            total = 0
            for rot in range(1, ROTATE_STEPS + 1):
                total += collect_rotation(s0, s1, motor, eid, rot)
                print(f"\n  Rotation {rot} done. Total saved so far: {total}")
            print(f"\n  [{eid}] complete. Total saved: {total}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        try:
            motor.set_drive(0,0,0); motor.set_drive(1,0,0); motor.disable()
            select_mux(SENSOR_0_PORT); s0.disable_bulb(0)
            select_mux(SENSOR_1_PORT); s1.disable_bulb(0)
            select_mux('off')
        except Exception as e:
            print(f"Cleanup error: {e}")
        print("Exited.")


if __name__ == '__main__':
    main()
