import time
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch
from numpy import VisibleDeprecationWarning
import warnings
import os

warnings.filterwarnings('error')


def collate_func(input_x):
    image_set = []
    label_set = []
    for images, labels in input_x:
        image_set.append(images)
        label_set.append(labels)
    return image_set, label_set


def face_detection(root_path, name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(margin=20, keep_all=False, device=device)
    folder = ImageFolder(f'{root_path}/{name}', transform=transforms.Resize((480, 640)))
    if name == '':
        old_file_list = []
    else:
        try:
            old_file_list = os.listdir(f'{root_path + "_cropped"}/{name}/{name}')
        except:
            old_file_list = []
    folder.samples = [(i, i.replace(root_path, root_path + '_cropped')) for i, _ in folder.samples if os.path.basename(i) not in old_file_list]
    loader = DataLoader(folder, batch_size=16, collate_fn=collate_func)
    with tqdm(total=len(folder), desc='Detection', unit='img') as bar:
        for img_list, path in loader:
            try:
                mtcnn(img_list, save_path=path)
            except VisibleDeprecationWarning:
                for img, p in zip(img_list, path):
                    mtcnn(img, save_path=p)
            bar.update(len(img_list))
    return old_file_list
