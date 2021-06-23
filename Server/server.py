import datetime
import os
from socket import *
from threading import Thread
import demjson
from FaceRecognition import register_new, model
from mysql import Sql

sql = Sql()


class UbuntuServer:
    def __init__(self, address_port):
        self.s = socket(AF_INET, SOCK_STREAM)
        self.s.bind(address_port)
        self.s.listen(5)

    def serve_forever(self):
        while True:
            client, address = self.s.accept()
            threads = Thread(target=Handler(client, address).handle)
            threads.setDaemon(True)
            threads.start()


class Handler:
    def __init__(self, client, address):
        self.client = client
        self.address = address
        print('Hello:', address)

    def _get_command(self):
        try:
            length = int(self.client.recv(4)[1:])
            return demjson.decode(self.client.recv(length))
        except:
            return {}

    def _send_result(self, result):
        byte = demjson.encode(result).encode('utf8')
        length = len(byte)
        if length < 10:
            prefix = f'#00{length}'
        elif length < 100:
            prefix = f'#0{length}'
        else:
            prefix = f'#{length}'
        self.client.sendall(prefix.encode('utf8') + byte)

    def handle(self):
        next_time = datetime.datetime.now() + datetime.timedelta(minutes=1)
        first = True
        while True:
            command_dict = self._get_command()
            command = command_dict.get('command')
            command = command if command is not None else 'close'

            if command == 'register':
                name = command_dict.get('name')
                try:
                    os.mkdir(f'dataset/{name}')
                    os.mkdir(f'dataset/{name}/{name}')
                except:
                    pass
                if self._get_command().get('tag') == 'finished':
                    register_new(name)
                    self._send_result({'status': 'OK'})

            elif command == 'test':
                frame_path = command_dict.get('frame_path')
                if self._get_command().get('tag') == 'finished':
                    model.load_model()
                    identities = model.test_face(frame_path)
                    self._send_result({'identities': identities})

            elif command == 'status':
                delta_time = datetime.datetime.now() - next_time
                client_status = command_dict.get('status')
                web_status = sql.get_status()
                new_status = {'Time': client_status.get('Time')}
                return_status = {}
                flag = True
                for k1, k2 in zip(['Door', 'Fan', 'Wetter', 'Curtain'],
                                  ['ctrl_door', 'ctrl_fan', 'ctrl_wetter', 'ctrl_curtain']):
                    if web_status.get(k2) == 1:
                        new_status[k1] = web_status.get(k1)
                        new_status[k2] = 1
                        return_status[k1] = web_status.get(k1)
                    else:
                        new_status[k1] = client_status.get(k1)
                        new_status[k2] = 0
                        return_status[k1] = None
                    flag = flag and new_status[k1] == web_status[k1]
                if first:
                    sql.insert_status(client_status)
                    sql.insert_device_status(new_status)
                    self._send_result({'new_status': new_status})
                    first = False
                    continue
                if not flag:
                    sql.update_status(new_status)
                if delta_time.seconds >= 0 and delta_time.days == 0:
                    sql.insert_status(client_status)
                    next_time = datetime.datetime.now() + datetime.timedelta(minutes=1)
                self._send_result({'new_status': return_status})

            elif command == 'close':
                self.client.close()
                print('Bye:', self.address)
                break
            else:
                pass


if __name__ == '__main__':
    UbuntuServer(('localhost', 8025)).serve_forever()
