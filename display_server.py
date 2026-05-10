# display_server.py
# Run independently: python3 display_server.py
# Receives messages from main.py over a local socket and renders the TFT display.

import os
os.environ.setdefault("DISPLAY", ":0")
os.environ["SDL_VIDEODRIVER"]       = "x11"
os.environ["SDL_RENDERDRIVER"]      = "software"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

import pygame
pygame.init()

import socket
import threading
import time

# Import your existing display internals (everything except the Dashboard class)
from src.display_core import (
    W, H, FULLSCREEN, C, FPS,
    BootScreen, DashScreen
)

HOST = "127.0.0.1"
PORT = 65432


def parse_message(msg):
    """Parse 'phase|status0|status1' into a state dict."""
    parts = msg.strip().split("|")
    if len(parts) == 3:
        return {"phase": parts[0], "status_0": parts[1], "status_1": parts[2]}
    elif len(parts) == 2:
        return {"phase": parts[0], "status_0": parts[1], "status_1": parts[1]}
    return None


def socket_listener(state):
    """Listens for messages from main.py in a background thread."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, PORT))
        srv.listen(1)
        srv.settimeout(1.0)
        print(f"[display] Listening on {HOST}:{PORT}")

        while state.get("running", True):
            try:
                conn, _ = srv.accept()
                with conn:
                    while state.get("running", True):
                        try:
                            data = conn.recv(256)
                            if not data:
                                break
                            msg = data.decode().strip()
                            if msg == "QUIT":
                                state["running"] = False
                                break
                            parsed = parse_message(msg)
                            if parsed:
                                state.update(parsed)
                        except (ConnectionResetError, OSError):
                            break
            except socket.timeout:
                continue


def main():
    if FULLSCREEN:
        try:
            screen = pygame.display.set_mode((W, H), pygame.FULLSCREEN)
        except Exception:
            screen = pygame.display.set_mode((W, H))
    else:
        screen = pygame.display.set_mode((W, H))

    pygame.display.set_caption("Talong AI")
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()

    def fnt(size, bold=False):
        for name in ["Courier New", "Consolas", "DejaVu Sans Mono",
                     "Lucida Console", "monospace"]:
            try:
                return pygame.font.SysFont(name, size, bold=bold)
            except Exception:
                pass
        return pygame.font.Font(None, size)

    fs = W / 800.0
    fonts = {
        "boot_title" : fnt(int(46*fs), bold=True),
        "boot_sub"   : fnt(int(16*fs)),
        "boot_log"   : fnt(int(15*fs)),
        "bar_title"  : fnt(int(20*fs), bold=True),
        "panel_title": fnt(int(20*fs), bold=True),
        "status"     : fnt(int(30*fs), bold=True),
        "sub"        : fnt(int(16*fs)),
        "tiny"       : fnt(int(14*fs)),
        "key"        : fnt(int(17*fs), bold=True),
    }

    state = {
        "running"  : True,
        "phase"    : "BOOTING",
        "status_0" : "LOADING AI...",
        "status_1" : "LOADING AI...",
    }

    # Socket listener runs in background — pygame owns the main thread
    listener = threading.Thread(target=socket_listener, args=(state,), daemon=True)
    listener.start()

    boot      = BootScreen(screen, fonts)
    dash      = None
    scr_state = "boot"

    while state["running"]:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                state["running"] = False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    state["running"] = False
                if ev.key == pygame.K_SPACE and scr_state == "boot":
                    boot._fo   = True
                    boot._fo_t = time.time()

        if scr_state == "boot":
            boot.update()
            boot.draw()
            if boot.done:
                scr_state = "dash"
                dash = DashScreen(screen, fonts, state)
        else:
            dash.update()
            dash.draw()

        pygame.display.flip()
        clock.tick(FPS)

    screen.fill((0, 0, 0))
    pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
