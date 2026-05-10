# data_collection/collect.py
"""
Data collection with per-segment labeling for infested eggplants.
Run directly: python -m data_collection.collect
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
    PASSES_PER_SIDE_HEALTHY, PASSES_PER_SIDE_INFESTED, ROTATE_STEPS,
)

bus = smbus2.SMBus(I2C_BUS)


# ── Hardware helpers ──────────────────────────────────────────────────────────

def select_mux(ch):
    bus.write_byte(MUX_ADDR, 0x00 if ch == 'off' else 1 << ch)

def init_sensor(channel):
    select_mux(channel)
    time.sleep(0.5)
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
            readings = [
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
            return readings
        except OSError:
            try: sensor.disable_bulb(0)
            except: pass
            time.sleep(0.2)
    print("  [Warning] Sensor busy, skipping.")
    return None

def motor_forward(motor):
    motor.set_drive(0, 0, MOTOR_SPEED)
    motor.set_drive(1, 1, MOTOR_SPEED)
    time.sleep(MOVE_DURATION)
    motor.set_drive(0, 0, 0)
    motor.set_drive(1, 0, 0)
    time.sleep(0.2)

def motor_home(motor):
    motor.set_drive(0, 1, MOTOR_SPEED)
    motor.set_drive(1, 0, MOTOR_SPEED)
    time.sleep(RETURN_DURATION)
    motor.set_drive(0, 0, MOTOR_SPEED)
    motor.set_drive(1, 1, MOTOR_SPEED)
    time.sleep(0.5)
    motor.set_drive(0, 0, 0)
    motor.set_drive(1, 0, 0)


# ── CSV helpers ───────────────────────────────────────────────────────────────

def ensure_csv():
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
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
    Asks which of the 3 conveyor segments face the infested area.
    Returns a set of 0-indexed segment numbers, e.g. {0, 1}.
    'all' → {0,1,2},  'none' → {}
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

def run_pass(sensor0, sensor1, motor, eggplant_id,
             label, infested_segments=None) -> dict:
    """
    One full conveyor pass (3 segments).
    Healthy: all segments saved as 'Healthy'.
    Infested: only segments in infested_segments saved; others discarded.
    """
    saved = {'Healthy': 0, 'Infested': 0, 'Discarded': 0}

    for step in range(3):
        r0 = get_readings(sensor0, SENSOR_0_PORT)
        r1 = get_readings(sensor1, SENSOR_1_PORT)

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
            motor_forward(motor)

    motor_home(motor)
    return saved


# ── Per-eggplant collection ───────────────────────────────────────────────────

def collect_eggplant(sensor0, sensor1, motor, label, eggplant_id):
    total = {'Healthy': 0, 'Infested': 0, 'Discarded': 0}

    for rot in range(ROTATE_STEPS):
        print(f"\n  Rotation {rot+1}/{ROTATE_STEPS}")
        if rot > 0:
            input("  Rotate eggplant ~45° then press [ENTER]: ")

        inf_segs = None
        if label == 'Infested':
            print("  Segment layout: [1]=front  [2]=middle  [3]=back")
            inf_segs = ask_infested_segments()
            if not inf_segs:
                print("  No infested segments — skipping rotation.")
                motor_forward(motor); motor_forward(motor); motor_home(motor)
                continue

        passes = PASSES_PER_SIDE_INFESTED if label == 'Infested' else PASSES_PER_SIDE_HEALTHY
        for p in range(passes):
            print(f"    Pass {p+1}/{passes}...", end=' ')
            saved = run_pass(sensor0, sensor1, motor, eggplant_id, label, inf_segs)
            for k in total: total[k] += saved.get(k, 0)
            print(f"I:{saved.get('Infested',0)}  H:{saved.get('Healthy',0)}  "
                  f"discarded:{saved.get('Discarded',0)}")

    print(f"\n  [{eggplant_id}] done — "
          f"I:{total['Infested']}  H:{total['Healthy']}  "
          f"discarded:{total['Discarded']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== EGGPLANT DATA COLLECTION ===\n")

    sensor0 = init_sensor(SENSOR_0_PORT)
    sensor1 = init_sensor(SENSOR_1_PORT)
    if not sensor0 or not sensor1:
        print("Sensor failure. Exiting."); return

    motor = qwiic_scmd.QwiicScmd()
    if not motor.connected:
        print("Motor driver not found. Exiting."); return
    motor.begin(); motor.enable()

    print("Homing conveyor...")
    motor_home(motor)

    print("\nSensor warm-up (2 min). Place eggplant and close box.")
    time.sleep(120)
    print("Warm-up done.\n")

    ensure_csv()

    try:
        while True:
            print("\n" + "="*50)
            choice = input("Label  H=Healthy / I=Infested / Q=quit: ").strip().upper()
            if choice == 'Q': break
            if choice not in ('H', 'I'):
                print("  Enter H, I, or Q."); continue

            label      = 'Healthy' if choice == 'H' else 'Infested'
            eggplant_id = input("Eggplant ID (e.g. H01, I03): ").strip().upper()
            if not eggplant_id:
                print("  ID cannot be empty."); continue

            input(f"  Place eggplant, close box, press [ENTER]: ")
            collect_eggplant(sensor0, sensor1, motor, label, eggplant_id)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        try:
            motor.set_drive(0,0,0); motor.set_drive(1,0,0); motor.disable()
            select_mux(SENSOR_0_PORT); sensor0.disable_bulb(0)
            select_mux(SENSOR_1_PORT); sensor1.disable_bulb(0)
            select_mux('off')
        except Exception as e:
            print(f"Cleanup error: {e}")
        print("Exited.")


if __name__ == '__main__':
    main()
