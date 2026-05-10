# macro_listener.py
"""
Macro pad listener — runs in the 'talongs_keys' tmux session.

Power state is persisted in STATE_FILE (JSON) so it survives reboots.

Boot behaviour
──────────────
  On every boot, this script reads state.json.
  - state == 'off'  →  blank_screen.py is launched (black display),
                        waiting for the user to press button 1.
  - state == 'on'   →  main.py is launched automatically
                        (handles the case where the Pi rebooted mid-session).

Key mapping
───────────
  KEY_A  (1)  Power toggle
                off → on  : launch main.py, state = 'on'
                on  → off : gracefully quit main.py, black screen 1 s,
                            state = 'off', then  sudo shutdown now
  KEY_B  (2)  SCAN        — send [ENTER] to trigger a scan
  KEY_C  (3)  CANCEL scan  — abort current scan, return home, go to READY
  KEY_D  (4)  RESET        — graceful quit then relaunch
"""

import json
import os
import time
import subprocess

try:
    import evdev
    from evdev import ecodes
except ImportError:
    print("ERROR: 'evdev' is not installed.")
    print("Fix: micromamba run -n talong_stable pip install evdev")
    raise SystemExit(1)


# ── Config ─────────────────────────────────────────────────────────────────────
TMUX_SESSION = 'talongs'
MAIN_SCRIPT  = 'python3 main.py'
BLANK_SCRIPT = 'python3 blank_screen.py'
STATE_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'state.json')


# ── State helpers ──────────────────────────────────────────────────────────────

def read_state() -> str:
    """Return 'on' or 'off'. Defaults to 'off' if file missing or corrupt."""
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f).get('power', 'off')
    except Exception:
        return 'off'


def write_state(power: str):
    """Write {'power': 'on'|'off'} to STATE_FILE."""
    with open(STATE_FILE, 'w') as f:
        json.dump({'power': power}, f, indent=2)
    print(f"[state] power = {power}")


# ── tmux helpers ───────────────────────────────────────────────────────────────

def send(*args):
    """Send key(s) to the main tmux session."""
    subprocess.run(['tmux', 'send-keys', '-t', TMUX_SESSION, *args])


def tmux_run(cmd: str):
    """Replace whatever is running in the tmux session with a new command."""
    # Send Ctrl-C first in case something is running, then run the new command
    send('C-c')
    time.sleep(0.5)
    send(cmd, 'C-m')


def launch_blank():
    """Show the black standby screen."""
    tmux_run(BLANK_SCRIPT)


def launch_main():
    """Start the TalongAI main program."""
    tmux_run(MAIN_SCRIPT)


def quit_main():
    """Gracefully quit main.py (sends 'q' then Enter)."""
    send('q', 'C-m')
    time.sleep(2)   # give main.py time to clean up hardware


def shutdown():
    """Black screen for 1 second then power off."""
    launch_blank()
    time.sleep(1)
    subprocess.run(['sudo', 'shutdown', 'now'])


# ── Keyboard detection ─────────────────────────────────────────────────────────

def find_keyboard():
    """Return the first evdev device that has both KEY_A and KEY_B."""
    for path in evdev.list_devices():
        try:
            dev  = evdev.InputDevice(path)
            caps = dev.capabilities()
            if ecodes.EV_KEY in caps:
                keys = caps[ecodes.EV_KEY]
                if ecodes.KEY_A in keys and ecodes.KEY_B in keys:
                    return dev
        except Exception:
            continue
    return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    keyboard = None
    try:
        keyboard = evdev.InputDevice('/dev/input/by-id/usb-1189_8890-event-kbd')
    except (FileNotFoundError, PermissionError):
        keyboard = find_keyboard()

    if keyboard is None:
        print("Macro keyboard not found. Is it plugged in?")
        return

    print(f"Listening to: {keyboard.name}")

    # ── Restore state on boot ──────────────────────────────────────────────────
    power = read_state()
    print(f"[boot] Restored state: power = {power}")

    if power == 'on':
        # Pi may have rebooted mid-session — relaunch the app
        print("[boot] Relaunching main.py...")
        launch_main()
    else:
        # Show black standby screen
        print("[boot] Showing blank screen (device is off).")
        launch_blank()

    # ── Event loop ─────────────────────────────────────────────────────────────
    try:
        for event in keyboard.read_loop():
            if event.type != ecodes.EV_KEY:
                continue
            key = evdev.categorize(event)
            if key.keystate != key.key_down:
                continue

            # ── Button 1 — Power toggle ────────────────────────────────────────
            if key.keycode == 'KEY_A':
                power = read_state()        # always read fresh from disk
                if power == 'off':
                    print("[KEY_A] OFF → ON: launching main.py")
                    write_state('on')
                    launch_main()
                else:
                    print("[KEY_A] ON → OFF: shutting down")
                    write_state('off')
                    quit_main()
                    shutdown()              # shows blank screen then powers off

            # ── Button 2 — Trigger scan ────────────────────────────────────────
            elif key.keycode == 'KEY_B':
                if read_state() == 'on':
                    send('C-m')

            # ── Button 3 — Cancel scan, return home, go back to READY ────────────
            elif key.keycode == 'KEY_C':
                if read_state() == 'on':
                    print("[KEY_C] Cancel scan — sending 'c' to main.py")
                    send('c', 'C-m')   # main.py picks this up as 'cancel'

            # ── Button 4 — Graceful reset ──────────────────────────────────────
            elif key.keycode == 'KEY_D':
                if read_state() == 'on':
                    quit_main()
                    launch_main()

    except KeyboardInterrupt:
        print("Listener stopped.")


if __name__ == '__main__':
    main()
