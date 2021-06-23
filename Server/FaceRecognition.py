from datetime import datetime
import torch
from PIL import ImageDraw, Image
import cv2
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from facenet_pytorch import fixed_image_standardization, MTCNN
import numpy as np
from model.InceptionResNetV1 import InceptionResNetV1
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import os

dataset = 'dataset'
dataset_cropped = 'dataset_cropped'
model_path = 'model/model_path'
detected_faces = 'detected_save'


class FaceRecognition:
    def __init__(self, pretrained_dataset='casia-webface', load=None, cuda=True):
        self.pretrained_dataset = pretrained_dataset
        if cuda and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        print('using device:', self.device)
        self.resnet = InceptionResNetV1(pretrained=self.pretrained_dataset).to(self.device).eval()
        self.mtcnn = MTCNN(margin=20, keep_all=True, device=self.device).eval()
        self.classifier = SVC(probability=True)
        self.load = load
        if load is not None:
            self.load_model(load)
        self.count = len(os.listdir('/home/kepler/Desktop/djangoProject/static/web_5/')) + 1

    def _load_dataset(self, dataset_path, batch_size, incremental=False, repeated=False, new_id=None,
                      old_file_list=None):
        tf = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization
        ])
        data_folder = datasets.ImageFolder(dataset_path, transform=tf)
        if incremental:
            key = list(data_folder.class_to_idx.keys())[0]
            data_folder.class_to_idx[key] = new_id
            if not repeated:
                self.temp_id_to_name = {value: key for key, value in data_folder.class_to_idx.items()}
                data_folder.samples = [
                    (image, new_id) for image, _ in data_folder.samples
                ]
            else:
                self.temp_id_to_name = {}
                data_folder.samples = [
                    (image, new_id) for image, _ in data_folder.samples
                    if os.path.basename(image) not in old_file_list
                ]
        else:
            self.temp_id_to_name = {value: key for key, value in data_folder.class_to_idx.items()}
        data_loader = DataLoader(data_folder, batch_size=batch_size, shuffle=True)
        return data_loader, len(data_folder)

    def generate_embeddings(self, batch_size,
                            dataset_path=dataset_cropped,
                            load_path=model_path,
                            incremental_learning=False,
                            new_name=None,
                            old_file_list=[]
                            ):
        repeated = False
        if incremental_learning:
            if len(old_file_list) == 0:
                new_id = len(datasets.ImageFolder(dataset_cropped).class_to_idx) - 1
            else:
                repeated = True
                new_id = [k for k, v in self.id_to_name.items() if v == new_name][0]
        else:
            new_id = None

        # load dataset
        data_loader, dataset_len = self._load_dataset(dataset_path, batch_size,
                                                      incremental=incremental_learning,
                                                      repeated=repeated, new_id=new_id,
                                                      old_file_list=old_file_list
                                                      )

        # whether using old embeddings
        if incremental_learning:
            # self.load_model(load_path)
            self.id_to_name.update(self.temp_id_to_name)
        else:
            self.embeddings = torch.Tensor().to(self.device)
            self.ids = torch.Tensor().to(self.device)
            self.id_to_name = self.temp_id_to_name
            self.threshold = torch.Tensor([0.5])

        with torch.no_grad():
            with tqdm(total=dataset_len, desc='Generating embeddings') as bar:
                for image, label in data_loader:
                    image = image.to(self.device)
                    label = label.to(self.device)
                    sub_embedding = self.resnet(image)
                    self.embeddings = torch.cat((self.embeddings, sub_embedding), 0)
                    self.ids = torch.cat((self.ids, label), 0)
                    bar.update(image.size()[0])

    def build_classifier(self):
        data = self.embeddings.cpu().numpy()
        label = self.ids.cpu().numpy().astype(np.int8)
        #
        # fine-tune SVC for better threshold
        # train_x, test_x, train_y, test_y = train_test_split(dataset, label, test_size=0.2)
        # svm = SVC(probability=True)
        # svm.fit(train_x, train_y)
        # print(svm.score(test_x, test_y))

        self.classifier.fit(data, label)

    def load_model(self, root_path=model_path):
        try:
            self.embeddings = torch.load(root_path + '/embeddings.pt').to(self.device)
            self.ids = torch.load(root_path + '/ids.pt').to(self.device)
            self.id_to_name = torch.load(root_path + '/id_to_name.pt')
            self.threshold = torch.load(root_path + '/threshold.pt')
            with open(root_path + '/classifier.pickle', 'rb') as f:
                self.classifier = pickle.load(f)
        except:
            raise Exception('Loading model error')

    def save_model(self, root_path=model_path):
        torch.save(self.embeddings.cpu(), root_path + '/embeddings.pt')
        torch.save(self.ids.cpu(), root_path + '/ids.pt')
        torch.save(self.id_to_name, root_path + '/id_to_name.pt')
        torch.save(self.threshold, root_path + '/threshold.pt')
        with open(root_path + '/classifier.pickle', 'wb') as f:
            pickle.dump(self.classifier, f)

    def test_face(self, frame_path):
        frame = Image.open(frame_path)
        boxes, probabilities = self.mtcnn.detect(frame)
        if boxes is None:
            frame.save(frame_path.replace('test', 'drawn'))
            return []
        faces_num = boxes.shape[0]
        faces = self.mtcnn(frame)
        identities = []
        any_unknown = False
        with torch.no_grad():
            sub_embedding = self.resnet(faces.to(self.device))
            probability = self.classifier.predict_proba(sub_embedding.cpu())
            prediction = np.argmax(probability, axis=1)
            for i, prob, pre in zip(range(faces_num), probability, prediction):
                # print(prob.max(), pre)
                # print(prob[pre], prob)
                if prob[pre] < .81 or pre == 0:
                    identity = 'Unknown'
                    any_unknown = True
                    color = 'red'
                else:
                    identity = self.id_to_name[pre]
                    color = 'green'
                print(prob[pre])
                frame = draw_faces(frame, boxes[i], identity, box_color=color)
                identities.append(identity)
        if any_unknown:
            frame.save(frame_path.replace('test', 'Unknown'))
            web_5_dir = '/home/kepler/Desktop/djangoProject/static/web_5/'
            new_name = self.count % 2
            self.count += 1
            frame.save(f'{web_5_dir}{new_name}.jpg')
        frame.save(frame_path.replace('test', 'drawn'))
        return identities


def first_train():
    """
    first-train template
    batch_size = 32
    incremental_learning = False
    """
    from FaceDetection import face_detection
    face_detection(dataset, '')
    model.load = None
    model.generate_embeddings(batch_size=16, incremental_learning=False)
    model.build_classifier()
    model.save_model()


def register_new(name, if_collect=True, host=False, size=1000):
    """
    register new person
    load = 'model/model_path'
    batch_size = 32
    incremental_learning = True
    """
    from FaceDetection import face_detection
    # from FaceCollection import face_collection
    # if if_collect:
    #     face_collection(size, 1, name, host)
    old_file_list = face_detection(dataset, name)
    model.load_model()
    model.generate_embeddings(dataset_path=f'{dataset_cropped}/{name}',
                              batch_size=32, incremental_learning=True, new_name=name,
                              old_file_list=old_file_list)
    model.build_classifier()
    model.save_model()


def draw_faces(frame, box, text, landmark=None, box_color='green'):
    draw = ImageDraw.Draw(frame)
    draw.rectangle(box, outline=box_color, width=4)
    draw.text(((box[0] + box[2]) / 2, box[3]), str(text), align='center')
    if landmark is not None:
        for ld in landmark:
            draw.ellipse([ld[0] - 4, ld[1] - 4, ld[0] + 4, ld[1] + 4], fill='yellow')
    return frame


def camera_trial():
    model.load_model()
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if success:
            frame = Image.fromarray(cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1))
            frame_path = f'detected_save/test/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.jpg'
            frame.save(frame_path)
            identities = model.test_face(frame_path)
            if len(identities) == 0:
                replace = 'test'
            else:
                replace = 'drawn'
            cv2.imshow('result',
                       cv2.cvtColor(np.array(Image.open(frame_path.replace('test', replace))),
                                    cv2.COLOR_BGR2RGB))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


def frame_trial(frame_path):
    model.load_model()
    model.test_face(frame_path)
    cv2.imshow('result',
               cv2.cvtColor(np.array(Image.open(frame_path.replace('test', 'drawn'))),
                            cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


model = FaceRecognition(load=None)

if __name__ == '__main__':
    # dir = os.listdir('detected_save/test/')
    # frame_trial('detected_save/test/' + dir[10])
    # camera_trial()
    # first_train()
    # register_new('your name', host=True)
    pass
