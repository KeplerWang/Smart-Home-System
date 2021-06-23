import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import cv2
from model.InceptionResNetV1 import InceptionResNetV1
from facenet_pytorch import MTCNN, training, fixed_image_standardization
from PIL import Image, ImageDraw


def draw_faces(frame, box, prob=None, landmark=None, result=None):
    draw = ImageDraw.Draw(frame)
    draw.rectangle(box, outline='red', width=6)
    if prob is not None:
        # draw.text(((box[0] + box[2]) / 2, box[3]), str(prob))
        draw.text(((box[0] + box[2]) / 2, box[3]), str(result))
    # if landmark is not None:
    #     for ld in landmark:
    #         draw.ellipse([ld[0] - 4, ld[1] - 4, ld[0] + 4, ld[1] + 4], fill='blue')
    return frame


device = torch.device('cpu')
names = torch.load("model/model_path/names.pt", map_location=device)
embeddings = torch.load("model/model_path/embeddings.pt", map_location=device)

resnet = InceptionResNetV1(pretrained='vggface2')
resnet.eval()
mtcnn = MTCNN(margin=20)
v_cap = cv2.VideoCapture(0)
while True:
    success, frame = v_cap.read()
    if not success:
        break
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes, probs, points = mtcnn.detect(frame, landmarks=True)
    if boxes is None:
        continue

    faces = mtcnn(frame)
    face_embedding = resnet(faces.unsqueeze(0).to(device))
    probability = [(face_embedding - embeddings[i]).norm().item() for i in range(embeddings.size()[0])]
    index = probability.index(min(probability))
    name = names[index]

    for box, prob, landmark in zip(boxes, probs, points):
        frame = draw_faces(frame, box, prob, landmark, (min(probability), name))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('1', np.array(frame, dtype=np.uint8))
v_cap.release()
