import qwiic_scmd
import time
import sys
import tty
import termios

# Initialize the Qwiic motor driver
myMotor = qwiic_scmd.QwiicScmd()

if myMotor.connected == False:
    print("Motor Driver not found. Check your Qwiic cables.")
    sys.exit()

myMotor.begin()
myMotor.enable()

# Function to read a single keypress instantly from the SSH terminal
def get_char():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

print("\n--- SSH Motor Control Active ---")
print(" Press 'f' -> Spin Forward")
print(" Press 'b' -> Spin Backward")
print(" Press SPACE -> Stop Motor")
print(" Press 'q' -> Quit program")
print("--------------------------------\n")

# Variables to track movement state and time
start_time = 0
current_direction = None

try:
    while True:
        char = get_char()
        
        if char == 'f':
            if current_direction != 'forward':
                # If we were going backward, log that time first before switching
                if current_direction is not None:
                    duration = time.time() - start_time
                    print(f"\rStopped {current_direction}. Ran for {duration:.2f} seconds.")
                
                start_time = time.time()
                current_direction = 'forward'
                print("\rSpinning Forward...      ", end="", flush=True)
                myMotor.set_drive(0, 0, 100)
                myMotor.set_drive(1, 0, 100)
                
        elif char == 'b':
            if current_direction != 'backward':
                # If we were going forward, log that time first before switching
                if current_direction is not None:
                    duration = time.time() - start_time
                    print(f"\rStopped {current_direction}. Ran for {duration:.2f} seconds.")
                
                start_time = time.time()
                current_direction = 'backward'
                print("\rSpinning Backward...     ", end="", flush=True)
                myMotor.set_drive(0, 1, 100)
                myMotor.set_drive(1, 1, 100)
                
        elif char == ' ': # Spacebar
            if current_direction is not None:
                duration = time.time() - start_time
                print(f"\rMotor Stopped. Ran {current_direction} for {duration:.2f} seconds.")
                current_direction = None
            else:
                # \r keeps the line clean if you spam the spacebar
                print("\rMotor is already stopped.                    ", end="", flush=True)
                
            myMotor.set_drive(0, 0, 0)
            myMotor.set_drive(1, 0, 0)
            
        elif char == 'q':
            # Log the final time if the motor was running when you pressed quit
            if current_direction is not None:
                duration = time.time() - start_time
                print(f"\rStopped {current_direction}. Ran for {duration:.2f} seconds.")
            print("\r\nQuitting...")
            break

except KeyboardInterrupt:
    pass
finally:
    # Safety first
    myMotor.set_drive(0, 0, 0)
    myMotor.set_drive(1, 0, 0)
    myMotor.disable()
    print("\r\nMotors disabled. Goodbye!")
