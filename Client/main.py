import time
import RPi.GPIO as GPIO
from hardware import Button, Buzzer, FireSensor, LightSensor, Relay, Servo, StepperMotor, TemperatureWithHumidity
from PiClient import PiClient
import threading
import random
import pyttsx3
import datetime

#####################################################################
# type                                  GPIO(BCM)                   #
# button for test_face                  21                          #
# button for register                   20                          #
# buzzer                                16                          #
# fire sensor                           12                          #
# light sensor                          5                           #
# relay for fan                         24                          #
# relay for wetter                      23                          #
# servo                                 27                          #
# stepper motor                         26, 19, 13, 6               #
# temperature with humidity             17(class specified)         #
# ip_addr   ip address and port negotiated                          #
#####################################################################

ip_addr = ('localhost', 8025)
engine = pyttsx3.init()
web_command = {}
current_status = {}
button_for_test_face = Button.Button(21)
button_for_register = Button.Button(20)
light_sensor = LightSensor.LightSensor(5)
stepper_motor = StepperMotor.StepperMotor(26, 19, 13, 6)
servo = Servo.Servo(27)
buzzer = Buzzer.Buzzer(16)
temp_with_hum = TemperatureWithHumidity.TemperatureWithHumidity()
wetter = Relay.Relay(23)
fan = Relay.Relay(24)
fire_sensor = FireSensor.FireSensor(12)


def thread_for_test_face():
    global ip_addr, engine, button_for_test_face, buzzer, web_command
    client = PiClient(ip_addr)
    while True:
        if button_for_test_face.get_button_status():
            result, identities = client.test_face()
            if result:
                servo.open()
                time.sleep(3)
                servo.close()
                for name in identities:
                    if name != 'Unknown':
                        engine.say('Welcome back home:' + name[:-4])
            else:
                buzzer.beep_batch(0.5, 3)
                engine.say('Warning！Strangers broke in!')
                engine.say('Warning！Strangers broke in!')
        elif web_command.get('Door') == 1:
            servo.open()
            # engine.say('Welcome! long time no see！')
            web_command['Door'] = None
        elif web_command.get('Door') == 0:
            servo.close()
        engine.runAndWait()


def thread_for_register():
    global ip_addr, engine, button_for_register
    client = PiClient(ip_addr)
    while True:
        if button_for_register.get_button_status():
            new_name = f'new{random.randint(100, 1000)}'
            if client.register(new_name, 500):
                engine.say(f'Register success! Your id is {new_name}.')
            else:
                engine.say('Register failed! Please try later.')
            engine.runAndWait()


def main_thread():
    global ip_addr, temp_with_hum, wetter, fan, stepper_motor, servo, web_command

    tftf = threading.Thread(target=thread_for_test_face)
    tftf.setDaemon(True)
    tftf.start()

    tfr = threading.Thread(target=thread_for_register)
    tfr.setDaemon(True)
    tfr.start()

    client = PiClient(ip_addr)
    while True:
        web_curtain = web_command.get('Curtain')
        command_curtain = web_curtain if web_curtain is not None else light_sensor.get_status()
        if command_curtain and stepper_motor.get_curtain_status() == 'down':
            stepper_motor.control('up')
        elif not command_curtain and stepper_motor.get_curtain_status() == 'up':
            stepper_motor.control('down')

        temperature, humidity = temp_with_hum.get_temperature_humidity()
        command_fan = web_command.get('Fan')
        command_fan = command_fan if command_fan is not None else temperature >= 26.0
        fan.open() if command_fan else fan.close()
        command_wetter = web_command.get('Wetter')
        # command_wetter = command_wetter if command_wetter is not None else humidity <= 50
        wetter.open() if command_wetter else wetter.close()

        buzzer.beep(0.75) if fire_sensor.get_fire_sensor_state() else None

        status = {
            'Temperature': temperature,
            'Humidity': humidity,
            'Wetter': wetter.get_status(),
            'Fan': fan.get_status(),
            'Curtain': stepper_motor.get_curtain_status() == 'up',
            'Door': servo.get_status(),
            'Time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        web_command = client.web(status)


if __name__ == '__main__':
    try:
        main_thread()
    except Exception as e:
        temp_with_hum.dht_close()
        engine.stop()
        GPIO.cleanup()
        raise
