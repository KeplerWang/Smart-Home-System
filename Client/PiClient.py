import time
from socket import *
from FaceCollection import face_collection
import demjson
import ftplib
from tqdm import tqdm
import cv2
from PIL import Image
from datetime import datetime


class PiClient:

    def __init__(self, server_addr):
        self.server_addr = server_addr
        self.s = socket(AF_INET, SOCK_STREAM)
        self.s.connect(server_addr)


    def test_face(self, if_fetch=False):
        """
        Steps:
        1. On client : upload picture (to detected) to server using FTP
        2. On server : do face recognition on remote server using trained model
        3. To client : return faces-drawn picture and recognition result to client

        During this process, SOCKET and FTP are enabled

        picture : picture to be detected    PIL.Image.Image
        drawn_picture : faces-drawn picture
        result : whether it's unauthorized faces
        """
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        while True:
            success, frame = cap.read()
            if success:
                frame = Image.fromarray(cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1))
                frame_path = f'detected_save/test/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.jpg'
                frame.save(frame_path)
                data = {'command': 'test', 'frame_path': frame_path}
                self._send_command(data)
                self._send_picture([frame_path])
                self._send_command({'tag': 'finished'})
                identities = self._get_result().get('identities')
                if if_fetch:
                    drawn_frame_path = frame_path.replace('test', 'drawn')
                    self._get_picture(drawn_frame_path)
                if len(identities) != 0:
                    return identities.count('Unknown') <= len(identities) / 2, identities
                else:
                    continue

    def register(self, name, size=500):
        """
        Steps:
        1. On client : collect 500 pictures of {name}
        2. On client : upload these pictures to server using FTP
        3. On server : do face detection on remote server and use CUDA accelerate
        4. On server : do Incremental Learning on remote server and
                       and generate new model(embeddings.pt, id_to_name.pt, ids_pt, classifier.pickle)
        5. To client : return ok to client

        During this process, SOCKET and FTP are enabled

        size : number of pictures
        millisecond : time(ms) delay of camera capture each picture
        name : name of the person to register
        host : specify whether to show camera or not when capturing. Keep True when you are in desktop mode
        """
        picture_names = face_collection(size=size, millisecond=1, name=name, host=False)
        data = {'command': 'register', 'name': name}
        self._send_command(data)
        time.sleep(0.5)
        self._send_picture(picture_names)
        self._send_command({'tag': 'finished'})
        return self._get_result().get('status') == 'OK'

    def web(self, status):
        data = {'command': 'status', 'status': status}
        self._send_command(data)
        return self._get_result().get('new_status')

    def close(self):
        self._send_command({'command': 'close'})

    def _send_command(self, command):
        byte = demjson.encode(command).encode('utf8')
        length = len(byte)
        if length < 10:
            prefix = f'#00{length}'
        elif length < 100:
            prefix = f'#0{length}'
        else:
            prefix = f'#{length}'
        self.s.sendall(prefix.encode('utf8') + byte)

    def _send_picture(self, picture_file):
        self.f = ftplib.FTP(self.server_addr[0])
        self.f.connect()
        self.f.login('username', 'password')
        with tqdm(total=len(picture_file), desc='Send to server', unit='img') as bar:
            for pic in picture_file:
                with open(pic, 'rb') as file:
                    self.f.storbinary('STOR /home/kepler/Desktop/FaceRecognitionServer/' + pic, file)
                bar.update()
        self.f.close()

    def _get_picture(self, picture_file):
        self.f = ftplib.FTP(self.server_addr[0])
        self.f.connect()
        self.f.login('username', 'password')
        with open(picture_file, 'wb') as file:
            self.f.retrbinary('RETR /home/kepler/Desktop/FaceRecognitionServer/' + picture_file, file.write)
        self.f.close()

    def _get_result(self):
        length = int(self.s.recv(4)[1:])
        return demjson.decode(self.s.recv(length))
