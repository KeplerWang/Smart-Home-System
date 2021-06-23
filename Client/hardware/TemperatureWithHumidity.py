import board
import adafruit_dht


class TemperatureWithHumidity:
    def __init__(self):
        self.dhtDevice = adafruit_dht.DHT22(board.D17)
        self.temperature_c = None
        self.humidity = None

    def get_temperature_humidity(self):
        count = 0
        while True:
            try:
                self.temperature_c = self.dhtDevice.temperature
                self.humidity = self.dhtDevice.humidity
                if self.temperature_c is not None and self.humidity is not  None:
                    break
            except:
                count += 1
                if count == 5 and self.temperature_c is not None and self.humidity is not None:
                    break
        return self.temperature_c, self.humidity

    def dht_close(self):
        self.dhtDevice.exit()


if __name__ == '__main__':
    x = TemperatureWithHumidity()
    print(x.get_temperature_humidity())
    x.dht_close()
