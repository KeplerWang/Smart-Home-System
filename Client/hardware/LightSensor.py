import RPi.GPIO as GPIO


class LightSensor:
    def __init__(self, pin=5):
        self.pin = pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    def get_status(self):
        return not GPIO.input(self.pin)


if __name__ == '__main__':
    # baseline of sensor - motor
    # down - 窗帘拉下状态   up - 窗帘抬起状态
    # True - 天亮         False - 天暗
    # down True -- > 窗帘将拉起
    # up False --> 窗帘将拉下
    # 其他保持不变
    from StepperMotor import StepperMotor
    sm = StepperMotor()
    ls = LightSensor()
    while True:
        # print(ls.get_status(), sm.get_curtain_status())
        if ls.get_status() and sm.get_curtain_status() == 'down':
            sm.control('up')
        elif not ls.get_status() and sm.get_curtain_status() == 'up':
            sm.control('down')




