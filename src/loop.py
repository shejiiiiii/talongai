# src/loop.py
"""
Hardware loop and IPC helpers — extracted from main.py.

Exports
-------
push()          — non-blocking IPC put to the display process queue
hardware_loop() — full scan lifecycle (init → home → warm-up → scan loop)
"""

import queue
import threading
import time

import smbus2

from config import (
    I2C_BUS, MUX_ADDR,
    SENSOR_0_PORT, SENSOR_1_PORT,
    MOTOR_SPEED, MOVE_DURATION, RETURN_DURATION,
    NUM_SEGMENTS,
)
from src.hardware import Multiplexer, SpectralSensor, Motor


# ─────────────────────────────────────────────────────────────────────────────
#  IPC helper
# ─────────────────────────────────────────────────────────────────────────────

def push(ipc_queue, **kwargs):
    """Non-blocking state-update put to the display process queue."""
    try:
        ipc_queue.put_nowait(kwargs)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  Stdin reader (background thread)
# ─────────────────────────────────────────────────────────────────────────────

def stdin_reader(cmd_queue: queue.Queue):
    """
    Reads stdin lines and forwards normalised commands to cmd_queue.

    ''   / anything else  →  'scan'
    'c'                   →  'cancel'
    'q'                   →  'quit'
    """
    while True:
        try:
            line = input().strip().lower()
            if line == 'q':
                cmd_queue.put('quit')
            elif line == 'c':
                cmd_queue.put('cancel')
            else:
                cmd_queue.put('scan')
        except EOFError:
            cmd_queue.put('quit')
            break


def wait_for_cmd(
    cmd_queue: queue.Queue,
    alive_fn,
    poll: float = 0.1,
) -> str:
    """Block until a command arrives or the display process dies."""
    while alive_fn():
        try:
            return cmd_queue.get(timeout=poll)
        except queue.Empty:
            continue
    return 'quit'


# ─────────────────────────────────────────────────────────────────────────────
#  Hardware loop
# ─────────────────────────────────────────────────────────────────────────────

def hardware_loop(ipc, analyzer, disp_proc):
    """
    Full hardware lifecycle: initialise → home → warm-up → scan loop.

    Parameters
    ----------
    ipc       : multiprocessing.Queue  — state updates pushed to display process
    analyzer  : PestAnalyzer | None    — None if model failed to load
    disp_proc : multiprocessing.Process
    """
    bus     = smbus2.SMBus(I2C_BUS)
    mux     = Multiplexer(bus, MUX_ADDR)
    motor   = None
    sensor0 = None
    sensor1 = None
    s0_ok = s1_ok = motor_ok = False

    push(ipc, phase='BOOTING', status_0='INIT...', status_1='INIT...')

    try:
        sensor0 = SpectralSensor(mux, SENSOR_0_PORT)
        s0_ok   = sensor0.initialize()
    except Exception as e:
        print(f"  Sensor 0 error: {e}")
    push(ipc, status_0='READY' if s0_ok else 'NOT CONNECTED')

    try:
        sensor1 = SpectralSensor(mux, SENSOR_1_PORT)
        s1_ok   = sensor1.initialize()
    except Exception as e:
        print(f"  Sensor 1 error: {e}")
    push(ipc, status_1='READY' if s1_ok else 'NOT CONNECTED')

    try:
        motor    = Motor()
        motor_ok = motor.initialize()
    except Exception as e:
        print(f"  Motor error: {e}")

    model_ok = analyzer is not None

    # ── Shared command queue ───────────────────────────────────────────────────
    cmd_queue: queue.Queue = queue.Queue()
    reader = threading.Thread(
        target=stdin_reader, args=(cmd_queue,), daemon=True
    )
    reader.start()

    # ── No hardware path ───────────────────────────────────────────────────────
    if not (s0_ok or s1_ok or motor_ok):
        phase = 'NO HARDWARE' if not model_ok else 'MODEL ONLY'
        push(ipc, phase=phase,
             status_0='NOT CONNECTED', status_1='NOT CONNECTED')
        print("\nNo hardware connected. Display running. Press [Q] + Enter to exit.\n")
        wait_for_cmd(cmd_queue, disp_proc.is_alive)
        push(ipc, running=False)
        return

    # ── Home ───────────────────────────────────────────────────────────────────
    if motor_ok:
        print("Homing conveyor...")
        push(ipc, phase='CALIBRATING', status_0='HOMING...', status_1='HOMING...')
        try:
            motor.return_home(MOTOR_SPEED, 2.0)
        except Exception as e:
            print(f"  Homing error: {e}")

    # ── Warm-up ────────────────────────────────────────────────────────────────
    print("Sensor warm-up (60 s). Place eggplant and close the box.")
    push(ipc, phase='STANDBY',
         status_0='WARMING UP' if s0_ok else 'NOT CONNECTED',
         status_1='WARMING UP' if s1_ok else 'NOT CONNECTED')
    time.sleep(60)
    print("Warm-up complete.\n")

    def go_ready():
        push(ipc, phase='STANDBY',
             status_0='SYSTEM READY' if s0_ok else 'NOT CONNECTED',
             status_1='SYSTEM READY' if s1_ok else 'NOT CONNECTED')

    go_ready()

    # ── Scan loop ──────────────────────────────────────────────────────────────
    try:
        while disp_proc.is_alive():
            if s0_ok: sensor0.enable_light()
            if s1_ok: sensor1.enable_light()

            print("\nPress [ENTER] to scan  |  [C] to cancel  |  [Q] to quit")
            cmd = wait_for_cmd(cmd_queue, disp_proc.is_alive)

            if s0_ok: sensor0.disable_light()
            if s1_ok: sensor1.disable_light()

            if cmd == 'quit':
                push(ipc, running=False)
                break

            if cmd == 'cancel':
                print("Cancel received (nothing was scanning).")
                go_ready()
                continue

            # ── Scan ──────────────────────────────────────────────────────────
            votes0, votes1 = [], []
            cancelled = False
            print("Starting conveyor scan...")

            for step in range(NUM_SEGMENTS):

                # Check for cancel at the start of each segment
                try:
                    incoming = cmd_queue.get_nowait()
                    if incoming in ('cancel', 'quit'):
                        print(
                            f"\n  [{'CANCEL' if incoming == 'cancel' else 'QUIT'}]"
                            f" received — aborting scan at segment {step + 1}."
                        )
                        cancelled = True
                        if incoming == 'quit':
                            push(ipc, running=False)
                            return
                        break
                except queue.Empty:
                    pass

                print(f"  Segment {step + 1}/{NUM_SEGMENTS}...")
                push(ipc,
                     phase=f'PART {step + 1}/{NUM_SEGMENTS}',
                     status_0='SCANNING...' if s0_ok else 'NOT CONNECTED',
                     status_1='SCANNING...' if s1_ok else 'NOT CONNECTED')

                if s0_ok and model_ok:
                    r0 = sensor0.read()
                    if r0:
                        lbl0, conf0, flip0 = analyzer.predict([r0])
                        votes0.append(lbl0)
                        print(
                            f"    [EGG 0]  {lbl0:<10} {conf0:.1f}%"
                            f"{'  [FLIPPED]' if flip0 else ''}"
                        )

                if s1_ok and model_ok:
                    r1 = sensor1.read()
                    if r1:
                        lbl1, conf1, flip1 = analyzer.predict([r1])
                        votes1.append(lbl1)
                        print(
                            f"    [EGG 1]  {lbl1:<10} {conf1:.1f}%"
                            f"{'  [FLIPPED]' if flip1 else ''}"
                        )

                if step < NUM_SEGMENTS - 1 and motor_ok:
                    push(ipc, phase='MOVING',
                         status_0='WAITING...', status_1='WAITING...')
                    motor.move_forward(MOTOR_SPEED, MOVE_DURATION)

            # Return home whether scan finished or was cancelled
            if motor_ok:
                print("Returning to home...")
                push(ipc, phase='RETURNING',
                     status_0='MOVING HOME', status_1='MOVING HOME')
                motor.return_home(MOTOR_SPEED, RETURN_DURATION)

            if cancelled:
                print("Scan cancelled. Back to READY.\n")
                go_ready()
                continue

            # ── Results ───────────────────────────────────────────────────────
            push(ipc, phase='AI ACTIVE',
                 status_0='PROCESSING...', status_1='PROCESSING...')

            final0, inf0, hlt0 = (
                analyzer.majority_vote(votes0) if votes0
                else ("NOT CONNECTED", 0, 0)
            )
            final1, inf1, hlt1 = (
                analyzer.majority_vote(votes1) if votes1
                else ("NOT CONNECTED", 0, 0)
            )

            print(f"\n{'='*50}")
            if votes0: print(f"  [EGG 0] {inf0}xI {hlt0}xH -> {final0.upper()}")
            if votes1: print(f"  [EGG 1] {inf1}xI {hlt1}xH -> {final1.upper()}")
            print(f"{'='*50}\n")

            push(ipc, phase='DONE',
                 status_0=final0.upper() if votes0 else 'NOT CONNECTED',
                 status_1=final1.upper() if votes1 else 'NOT CONNECTED')

    except KeyboardInterrupt:
        print("\nStopped by user.")
        push(ipc, running=False)

    finally:
        print("Cleaning up hardware...")
        try:
            if motor:   motor.disable()
        except Exception:
            pass
        try:
            if s0_ok and sensor0: sensor0.disable_light()
            if s1_ok and sensor1: sensor1.disable_light()
        except Exception:
            pass
        try:
            mux.select('off')
            bus.close()
        except Exception:
            pass
        print("Hardware cleaned up.")
        push(ipc, running=False)
