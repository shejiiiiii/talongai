# src/display_process.py
"""
Entry point for the spawned display process.

This file must NEVER import TensorFlow (directly or transitively).
It runs in a completely separate process started with spawn, so it
has a clean slate — no TF libraries, no GL context conflicts.
"""
import os


def run(queue):
    """
    Called by multiprocessing.Process in main.py.

    queue: multiprocessing.Queue
        Main process pushes state-update dicts here.
        e.g. queue.put({'phase': 'SCANNING', 'status_0': 'SCANNING...'})
        Push {'running': False} to signal shutdown.
    """
    os.environ.setdefault("DISPLAY", ":0")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

    import pygame
    # Import resolution/fullscreen constants from display module
    from src.display import W, H, FULLSCREEN
    pygame.init()

    if FULLSCREEN:
        try:
            screen = pygame.display.set_mode((W, H), pygame.FULLSCREEN)
        except Exception:
            screen = pygame.display.set_mode((W, H))
    else:
        screen = pygame.display.set_mode((W, H))

    pygame.display.set_caption("Talong AI")
    pygame.mouse.set_visible(False)
    print("[Display] Window opened.", flush=True)

    # Local state dict — updated by draining the queue each frame
    state = {
        "running"  : True,
        "phase"    : "BOOTING",
        "status_0" : "LOADING AI...",
        "status_1" : "LOADING AI...",
    }

    try:
        from src.display import _render_loop
        _render_loop(state, screen=screen, queue=queue)
    except Exception as exc:
        import traceback
        print("[Display] Fatal error:")
        traceback.print_exc()
    finally:
        try:
            pygame.quit()
        except Exception:
            pass
        print("[Display] Process exiting.", flush=True)
