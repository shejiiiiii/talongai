# src/hardware.py

import time
import smbus2
import qwiic_as7265x
import qwiic_scmd


class Multiplexer:
    """TCA9548A I2C multiplexer driver."""

    def __init__(self, bus: smbus2.SMBus, address: int):
        self._bus     = bus
        self._address = address

    def select(self, port):
        """Enable a single port (int 0-7). Pass 'off' to disable all channels."""
        if port == 'off':
            self._bus.write_byte(self._address, 0x00)
        else:
            self._bus.write_byte(self._address, 1 << int(port))


class SpectralSensor:
    """AS7265x 18-channel spectral sensor sitting behind a Multiplexer port."""

    # Calibrated-channel getter suffixes in order (A-L, R-W)
    _CHANNELS = ['a','b','c','d','e','f','g','h','i','j','k','l',
                 'r','s','t','u','v','w']

    def __init__(self, mux: Multiplexer, port: int):
        self._mux    = mux
        self._port   = port
        self._sensor = qwiic_as7265x.QwiicAS7265x()
        self.ready   = False

    def initialize(self) -> bool:
        """Connect and configure the sensor. Returns True on success."""
        print(f"  Initializing spectral sensor on port {self._port}...")
        self._mux.select(self._port)
        time.sleep(0.5)

        for _ in range(3):
            try:
                if self._sensor.begin():
                    self._sensor.soft_reset()
                    time.sleep(1)
                    self._sensor.set_gain(3)
                    self._sensor.set_integration_cycles(50)
                    self._sensor.disable_indicator()
                    self._sensor.set_bulb_current(12.5, 0)
                    self._sensor.disable_bulb(0)
                    print(f"  Sensor port {self._port} ready.")
                    self.ready = True
                    return True
            except OSError:
                time.sleep(0.5)

        print(f"  Error: Sensor port {self._port} not found.")
        return False

    def read(self) -> list | None:
        """
        Take one measurement and return 18 calibrated channel values (uW/cm²),
        or None if the sensor is unavailable or busy after 3 retries.
        """
        if not self.ready:
            return None

        for _ in range(3):
            try:
                self._mux.select(self._port)
                time.sleep(0.1)
                self._sensor.enable_bulb(0)
                time.sleep(0.1)
                self._sensor.take_measurements()
                readings = [
                    getattr(self._sensor, f'get_calibrated_{ch}')()
                    for ch in self._CHANNELS
                ]
                self._sensor.disable_bulb(0)
                return readings
            except OSError:
                try:
                    self._sensor.disable_bulb(0)
                except Exception:
                    pass
                time.sleep(0.2)

        print(f"  [Warning] Sensor port {self._port} busy — skipping scan.")
        return None

    def enable_light(self):
        """Turn the illumination bulb on (used while waiting for input)."""
        try:
            self._mux.select(self._port)
            time.sleep(0.05)
            self._sensor.enable_bulb(0)
        except Exception:
            pass

    def disable_light(self):
        """Turn the illumination bulb off."""
        try:
            self._mux.select(self._port)
            time.sleep(0.05)
            self._sensor.disable_bulb(0)
        except Exception:
            pass


class Motor:
    """SCMD-based dual DC motor controller for the conveyor belt."""

    def __init__(self):
        self._motor = qwiic_scmd.QwiicScmd()
        self.ready  = False

    def initialize(self) -> bool:
        """Connect and enable the motor driver. Returns True on success."""
        if not self._motor.connected:
            print("  Motor driver not found. Check connections.")
            return False
        self._motor.begin()
        self._motor.enable()
        print("  Motor driver ready.")
        self.ready = True
        return True

    def move_forward(self, speed: int, duration: float):
        """Drive the belt forward (toward sensors) for `duration` seconds."""
        self._motor.set_drive(0, 0, speed)
        self._motor.set_drive(1, 1, speed)
        time.sleep(duration)
        self._motor.set_drive(0, 0, 0)
        self._motor.set_drive(1, 0, 0)
        time.sleep(0.2)

    def return_home(self, speed: int, duration: float):
        """
        Drive the belt backward (home position) for `duration` seconds,
        then apply a brief brake pulse before full stop.
        """
        self._motor.set_drive(0, 1, speed)
        self._motor.set_drive(1, 0, speed)
        time.sleep(duration)
        # Brief braking pulse at low speed to avoid coasting past home
        self._motor.set_drive(0, 0, speed)
        self._motor.set_drive(1, 1, speed)
        time.sleep(0.5)
        self._motor.set_drive(0, 0, 0)
        self._motor.set_drive(1, 0, 0)

    def stop(self):
        """Immediately cut drive to both motors."""
        try:
            self._motor.set_drive(0, 0, 0)
            self._motor.set_drive(1, 0, 0)
        except Exception:
            pass

    def disable(self):
        """Stop and power-down the motor driver."""
        self.stop()
        try:
            self._motor.disable()
        except Exception:
            pass
