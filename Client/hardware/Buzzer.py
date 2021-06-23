import RPi.GPIO as GPIO
import time


class Buzzer:
    def __init__(self, pin=16):
        self.pin = pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)
        time.sleep(1)
        GPIO.output(self.pin, GPIO.LOW)

    def beep(self, seconds=0.5):
        GPIO.output(self.pin, GPIO.HIGH)
        time.sleep(seconds)
        GPIO.output(self.pin, GPIO.LOW)
        time.sleep(seconds)

    def beep_batch(self, seconds, counts):
        for i in range(counts):
            self.beep(seconds)


if __name__ == '__main__':
    b = Buzzer()
    # b.beep()
    # from gpiozero import Buzzer as Buzzer1
    # b = Buzzer1(pin=16, active_high=False, initial_value=True)