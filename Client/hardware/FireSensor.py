import time

import board
from adafruit_pcf8591.pcf8591 import PCF8591
import RPi.GPIO as GPIO


class FireSensor:
    def __init__(self, pin=12):
        self.pin = pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.IN)
        i2c = board.I2C()
        self.pcf = PCF8591(i2c, reference_voltage=5.0)

    def get_fire_sensor_state(self):
        if GPIO.input(self.pin) == 1:
            read_value = self.pcf.read(0)
            scaled_value = (read_value / 255) * self.pcf.reference_voltage
            return True
        else:
            return False


if __name__ == '__main__':
    from Buzzer import Buzzer
    buzzer = Buzzer(16)
    fire_sensor = FireSensor(12)
    while True:
        print(fire_sensor.get_fire_sensor_state())
        buzzer.beep(0.75) if fire_sensor.get_fire_sensor_state() else None
