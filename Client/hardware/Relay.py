import RPi.GPIO as GPIO
import time


class Relay:
    def __init__(self, pin):
        self.pin = pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)
        self.status = 0

    def open(self):
        GPIO.output(self.pin, GPIO.HIGH)
        self.status = 1

    def close(self):
        GPIO.output(self.pin, GPIO.LOW)
        self.status = 0

    def get_status(self):
        return self.status


if __name__ == '__main__':
    r = Relay(24)
    r.open()
    time.sleep(2)
    r.close()
    GPIO.cleanup()

