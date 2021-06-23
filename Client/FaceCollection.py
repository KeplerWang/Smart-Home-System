import time
import numpy as np
from datetime import datetime
import cv2
from PIL import Image
from tqdm import tqdm
import os


def face_collection(size, millisecond, name, host):
    try:
        os.mkdir(f'dataset/{name}')
        os.mkdir(f'dataset/{name}/{name}')
    except:
        pass
    path = f'dataset/{name}/{name}'
    frame_list = []
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    with tqdm(total=size, unit='img', desc='Images Collected') as bar:
        for i in range(size):
            success, frame = cap.read()
            if not success:
                raise Exception('reading camera error')
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_name = f'{path}/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}-{i+1}.jpg'
            frame.save(frame_name)
            frame_list.append(frame_name)
            bar.update()
            if host:
                cv2.imshow('1', cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB))
                cv2.waitKey(millisecond)
            else:
                time.sleep(millisecond/1000)
        cap.release()
        cv2.destroyAllWindows()
    return frame_list
