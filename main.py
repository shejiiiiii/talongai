# main.py
"""
TalongAI — entry point.

Process architecture (ARM/Pi: TF and pygame cannot share a process):
  Main process   → imports TF, runs hardware loop
  Display process → imports pygame, runs render loop (spawned fresh, no TF)

  IPC: multiprocessing.Queue  (main pushes state dicts, display reads them)

  spawn start method ensures the display child has zero TF state.

Commands (typed or sent via macro pad):
  [ENTER]  →  trigger a scan
  c        →  cancel current scan, return home, go back to READY
  q        →  quit the program
"""

import os
os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import time
import queue
import threading
from multiprocessing import Process, Queue as MPQueue
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ─────────────────────────────────────────────────────────────────────────────
#  IPC helper  —  push state updates to the display process
# ─────────────────────────────────────────────────────────────────────────────

def push(queue: MPQueue, **kwargs):
    """Non-blocking put; drops the update if the queue is full."""
    try:
        queue.put_nowait(kwargs)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  STDIN READER  —  runs in a background thread, feeds a command queue
# ─────────────────────────────────────────────────────────────────────────────

def stdin_reader(cmd_queue: queue.Queue):
    """
    Reads stdin lines in a background thread and puts them into cmd_queue.
    Recognised commands:
        ''   (empty / just Enter)  →  'scan'
        'c'                        →  'cancel'
        'q'                        →  'quit'
    """
    while True:
        try:
            line = input().strip().lower()
            if line == 'q':
                cmd_queue.put('quit')
            elif line == 'c':
                cmd_queue.put('cancel')
            else:
                cmd_queue.put('scan')   # Enter or anything else = scan
        except EOFError:
            cmd_queue.put('quit')
            break


def wait_for_cmd(cmd_queue: queue.Queue, alive_fn, poll: float = 0.1) -> str:
    """
    Block until a command arrives in cmd_queue or the display process dies.
    Returns the command string.
    """
    while alive_fn():
        try:
            return cmd_queue.get(timeout=poll)
        except queue.Empty:
            continue
    return 'quit'


# ─────────────────────────────────────────────────────────────────────────────
#  HARDWARE LOOP  (main process, background thread)
# ─────────────────────────────────────────────────────────────────────────────

def hardware_loop(ipc: MPQueue, analyzer, disp_proc: Process):
    from config import (
        I2C_BUS, MUX_ADDR,
        SENSOR_0_PORT, SENSOR_1_PORT,
        MOTOR_SPEED, MOVE_DURATION, RETURN_DURATION,
        NUM_SEGMENTS,
    )
    from src.hardware import Multiplexer, SpectralSensor, Motor
    from src.analyzer import PestAnalyzer

    import smbus2
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

    def alive():
        return disp_proc.is_alive()

    # ── Shared command queue (filled by stdin_reader thread) ──────────────────
    cmd_queue: queue.Queue = queue.Queue()
    reader = threading.Thread(target=stdin_reader, args=(cmd_queue,), daemon=True)
    reader.start()

    # ── No hardware path ───────────────────────────────────────────────────────
    if not (s0_ok or s1_ok or motor_ok):
        phase = 'NO HARDWARE' if not model_ok else 'MODEL ONLY'
        push(ipc, phase=phase,
             status_0='NOT CONNECTED', status_1='NOT CONNECTED')
        print("\nNo hardware connected. Display running. Press [Q] + Enter to exit.\n")
        cmd = wait_for_cmd(cmd_queue, alive)
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
        """Push READY state to the display."""
        push(ipc, phase='STANDBY',
             status_0='SYSTEM READY' if s0_ok else 'NOT CONNECTED',
             status_1='SYSTEM READY' if s1_ok else 'NOT CONNECTED')

    go_ready()

    # ── Scan loop ──────────────────────────────────────────────────────────────
    try:
        while alive():
            if s0_ok: sensor0.enable_light()
            if s1_ok: sensor1.enable_light()

            print("\nPress [ENTER] to scan  |  [C] to cancel  |  [Q] to quit")
            cmd = wait_for_cmd(cmd_queue, alive)

            if s0_ok: sensor0.disable_light()
            if s1_ok: sensor1.disable_light()

            if cmd == 'quit':
                push(ipc, running=False)
                break

            if cmd == 'cancel':
                # Nothing is scanning yet — just stay ready
                print("Cancel received (nothing was scanning).")
                go_ready()
                continue

            # ── cmd == 'scan' ──────────────────────────────────────────────────
            votes0, votes1 = [], []
            cancelled = False
            print("Starting conveyor scan...")

            for step in range(NUM_SEGMENTS):

                # ── Check for cancel at the start of each segment ──────────────
                try:
                    incoming = cmd_queue.get_nowait()
                    if incoming in ('cancel', 'quit'):
                        print(f"\n  [{'CANCEL' if incoming == 'cancel' else 'QUIT'}]"
                              f" received — aborting scan at segment {step + 1}.")
                        cancelled = True
                        if incoming == 'quit':
                            push(ipc, running=False)
                            return          # exit hardware_loop entirely
                        break
                except queue.Empty:
                    pass                    # no command — proceed normally

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
                        print(f"    [EGG 0]  {lbl0:<10} {conf0:.1f}%"
                              f"{'  [FLIPPED]' if flip0 else ''}")

                if s1_ok and model_ok:
                    r1 = sensor1.read()
                    if r1:
                        lbl1, conf1, flip1 = analyzer.predict([r1])
                        votes1.append(lbl1)
                        print(f"    [EGG 1]  {lbl1:<10} {conf1:.1f}%"
                              f"{'  [FLIPPED]' if flip1 else ''}")

                if step < NUM_SEGMENTS and motor_ok:
                    push(ipc, phase='MOVING',
                         status_0='WAITING...', status_1='WAITING...')
                    motor.move_forward(MOTOR_SPEED, MOVE_DURATION)

            # ── Return home (always — whether scan finished or was cancelled) ──
            if motor_ok:
                print("Returning to home...")
                push(ipc, phase='RETURNING',
                     status_0='MOVING HOME', status_1='MOVING HOME')
                motor.return_home(MOTOR_SPEED, RETURN_DURATION)

            # ── Cancelled mid-scan ─────────────────────────────────────────────
            if cancelled:
                print("Scan cancelled. Back to READY.\n")
                go_ready()
                continue

            # ── Normal finish — show results ───────────────────────────────────
            push(ipc, phase='AI ACTIVE',
                 status_0='PROCESSING...', status_1='PROCESSING...')

            final0, inf0, hlt0 = PestAnalyzer.majority_vote(votes0) \
                if votes0 else ("NOT CONNECTED", 0, 0)
            final1, inf1, hlt1 = PestAnalyzer.majority_vote(votes1) \
                if votes1 else ("NOT CONNECTED", 0, 0)

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
            if motor: motor.disable()
        except Exception: pass
        try:
            if s0_ok and sensor0: sensor0.disable_light()
            if s1_ok and sensor1: sensor1.disable_light()
        except Exception: pass
        try:
            mux.select('off')
            bus.close()
        except Exception: pass
        print("Hardware cleaned up.")
        push(ipc, running=False)


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    from src.display_process import run as display_run
    from src.analyzer import PestAnalyzer

    ipc_queue = MPQueue(maxsize=200)

    disp_proc = Process(
        target=display_run,
        args=(ipc_queue,),
        name="TalongDisplay",
        daemon=True,
    )
    disp_proc.start()
    print(f"Display process started (PID {disp_proc.pid}).")

    analyzer = None
    print("Loading AI model...")
    try:
        analyzer = PestAnalyzer()
        print("Model loaded OK.")
    except Exception as e:
        print(f"Model load failed: {e}")
        push(ipc_queue, status_0='MODEL FAIL', status_1='MODEL FAIL')

    hw_thread = threading.Thread(
        target=hardware_loop,
        args=(ipc_queue, analyzer, disp_proc),
        daemon=True,
        name="HardwareLoop",
    )
    hw_thread.start()

    disp_proc.join()
    print("Display closed.")
    push(ipc_queue, running=False)
    hw_thread.join(timeout=5)
    print("Exited.")


if __name__ == "__main__":
    main()
