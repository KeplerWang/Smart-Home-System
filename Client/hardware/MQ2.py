# import RPi.GPIO as GPIO
# import time
# import board
# import adafruit_pcf8591.pcf8591 as PCF
# from adafruit_pcf8591.analog_in import AnalogIn
# import math
#
# pin_fire=18
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(pin_fire, GPIO.IN, pull_up_down=GPIO.PUD_UP)
# i2c = board.I2C()
# pcf = PCF.PCF8591(i2c, address=0x48, reference_voltage=3.3)
# pcf_in_0 = AnalogIn(pcf, PCF.A0)
# LOAD_RESISTANCE = 5
# CO_CURVE = [2.30775, 0.71569, -0.33539]
#
# try:
#     while True:
#         status = GPIO.input(pin_fire)
#         if status == True:
#             print('没有检测到烟雾')
#             raw_value = pcf_in_0.value
#             scaled_value = (raw_value / 65535) * pcf_in_0.reference_voltage
#             resistance=None
#             resistance = resistance if resistance else LOAD_RESISTANCE
#             calc_resis =float(resistance * (1023.0 - scaled_value) / float(scaled_value))
#             calc_percentage =math.pow(10,((math.log(resistance) - CO_CURVE[1]) / CO_CURVE[2]) + CO_CURVE[0])
#             print("P0:  %0.2fV" % (scaled_value))
#             print("ppm: %0.2f" % (calc_percentage))
#         else:
#             print('检测到有烟雾')
#             raw_value = pcf_in_0.value
#             scaled_value = (raw_value / 65535) * pcf_in_0.reference_voltage
#             resistance=None
#             resistance = resistance if resistance else LOAD_RESISTANCE
#             calc_resis =float(resistance * (1023.0 - scaled_value) / float(scaled_value))
#             calc_percentage =math.pow(10,((math.log(resistance) - CO_CURVE[1]) / CO_CURVE[2]) + CO_CURVE[0])
#             print("ppm: %0.2f" % (calc_percentage))
#         time.sleep(1)
#
# except KeyboardInterrupt:
#     GPIO.cleanup()
#