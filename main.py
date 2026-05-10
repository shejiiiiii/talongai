# main.py
"""
TalongAI — entry point.

Process architecture (ARM/Pi: TF and pygame cannot share a process):
  Main process    → imports TF, runs hardware loop
  Display process → imports pygame, runs render loop (spawned fresh, no TF)

  IPC: multiprocessing.Queue  (main pushes state dicts, display reads them)

Commands (typed or sent via macro pad):
  [ENTER]  →  trigger a scan
  c        →  cancel current scan, return home, go back to READY
  q        →  quit the program
"""

import os
# Must be set before TensorFlow or pygame are imported.
os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import threading
import warnings
import multiprocessing
from multiprocessing import Process, Queue as MPQueue

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def main():
    from src.display_process import run as display_run
    from src.analyzer import PestAnalyzer
    from src.loop import push, hardware_loop

    ipc_queue = MPQueue(maxsize=200)

    disp_proc = Process(
        target=display_run,
        args=(ipc_queue,),
        name="TalongDisplay",
        daemon=True,
    )
    disp_proc.start()
    print(f"Display process started (PID {disp_proc.pid}).")

    print("Loading AI model...")
    analyzer = None
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
    # spawn ensures the display child starts with zero TF state.
    # Must be called inside the __main__ guard, not at module level,
    # so it doesn't re-execute in child processes on Windows/macOS.
    multiprocessing.set_start_method("spawn", force=True)
    main()
