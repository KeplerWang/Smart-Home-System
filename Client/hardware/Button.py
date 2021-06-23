import time
import RPi.GPIO as GPIO
# from gpiozero import Button as _Button

class Button:
    def __init__(self, pin=21):
        self.pin = pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def get_button_status(self):
        time.sleep(0.15)
        return GPIO.input(self.pin) != 1

    def add_interrupt(self, callback):
        GPIO.add_event_detect(self.pin, GPIO.RISING, callback=callback, bouncetime=200)


if __name__ == '__main__':
    # b = _Button(21)
    # b.wait_for_active()
    # print('yes')
    pass