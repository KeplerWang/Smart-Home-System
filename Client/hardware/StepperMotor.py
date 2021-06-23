import RPi.GPIO as GPIO
import time


class StepperMotor:
    def __init__(self, in1=26, in2=19, in3=13, in4=6, curtain_status='down'):
        self._in1, self._in2, self._in3, self._in4 = in1, in2, in3, in4
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self._in1, GPIO.OUT)
        GPIO.setup(self._in2, GPIO.OUT)
        GPIO.setup(self._in3, GPIO.OUT)
        GPIO.setup(self._in4, GPIO.OUT)
        self._curtain_status = curtain_status

    def _set_step(self, h1, h2, h3, h4):
        GPIO.output(self._in1, h1)
        GPIO.output(self._in2, h2)
        GPIO.output(self._in3, h3)
        GPIO.output(self._in4, h4)

    def control(self, ctrl):
        delay, steps = 0.004, 512
        if ctrl == 'stop':
            self._set_step(0, 0, 0, 0)
            return
        elif ctrl == 'down':
            self._curtain_status = 'down'
            pos = [0, 1, 2, 3]
        elif ctrl == 'up':
            self._curtain_status = 'up'
            pos = [3, 2, 1, 0]
        else:
            return

        for _ in range(steps):
            for i in range(4):
                pos_zero = [0, 0, 0, 0]
                pos_zero[pos[i]] = 1
                self._set_step(*pos_zero)
                time.sleep(delay)

    def get_curtain_status(self):
        return self._curtain_status


if __name__ == '__main__':
    x = StepperMotor()
    x.control('down')
    print(x.get_curtain_status())
    # destroy()
