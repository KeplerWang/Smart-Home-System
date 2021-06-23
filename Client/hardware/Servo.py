import RPi.GPIO as GPIO
import time
import numpy as np


class Servo:
    def __init__(self, pin=27):
        self.pin = pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.pin, 50)
        self.status = 0

    def open(self):
        self.pwm.start(2.5)
        control = np.arange(3, 12.6, 0.5)
        for x in control:
            self.pwm.ChangeDutyCycle(x)
        self.status = 1

    def close(self):
        self.pwm.start(12.5)
        control = np.arange(12, 2.4, -0.5)
        for x in control:
            self.pwm.ChangeDutyCycle(x)
        self.status = 0

    def get_status(self):
        return self.status


if __name__ == '__main__':
    s = Servo()
    s.open()
    time.sleep(3)
    s.close()
    time.sleep(3)
    # GPIO.cleanup()
