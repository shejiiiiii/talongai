import time
import board
import busio
import adafruit_tca9548a
import adafruit_vl53l1x

# 1. Initialize the standard Raspberry Pi I2C bus
i2c = busio.I2C(board.SCL, board.SDA)

# 2. Initialize the Multiplexer using the TCA9548A library's PCA9546A class
mux = adafruit_tca9548a.PCA9546A(i2c, address=0x70)

# 3. Create a virtual I2C bus for Port 3 
# (Port 3 corresponds to the 0x08 command you used earlier)
sensor_bus = mux[3]

# 4. Initialize the VL53L1X sensor on that specific virtual bus
vl53 = adafruit_vl53l1x.VL53L1X(sensor_bus)

# 5. Set to Short Distance Mode
# 1 = Short (up to ~1.3m, highly accurate, better ambient light immunity)
# 2 = Long (up to ~4m, default)
vl53.distance_mode = 1

# Optional: Set timing budget (in milliseconds)
# Lower budget = faster reads but slightly less accurate. 50ms is good for short mode.
vl53.timing_budget = 50 

# Start firing the laser
vl53.start_ranging()

print("Starting VL53L1X in Short Distance Mode. Press Ctrl+C to exit.")
print("-" * 50)

try:
    while True:
        # Check if the sensor has a new reading available
        if vl53.data_ready:
            distance = vl53.distance
            
            # The sensor returns None if it fails to get a reading
            if distance is not None:
                print(f"Distance: {distance} cm")
            else:
                print("Distance: Out of range / No reading")
                
            # Clear the interrupt so the sensor can take the next reading
            vl53.clear_interrupt()
            
        time.sleep(0.1) # Brief pause to prevent spamming the console

except KeyboardInterrupt:
    # Safely turn off the laser when you exit the script
    vl53.stop_ranging()
    print("\nLaser stopped. Exiting...")
