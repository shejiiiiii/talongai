import evdev

# Replace with your actual device path from 'ls /dev/input/by-id/'
device = evdev.InputDevice('/dev/input/by-id/usb-1189_8890-event-kbd')

print("Press your macro keys now... (Ctrl+C to stop)")
for event in device.read_loop():
    if event.type == evdev.ecodes.EV_KEY:
        key_event = evdev.categorize(event)
        if key_event.keystate == 1: # Key Down
            print(f"Detected: {key_event.keycode}")
