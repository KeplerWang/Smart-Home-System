import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
from model.InceptionResNetV1 import InceptionResNetV1
from facenet_pytorch import MTCNN, training, fixed_image_standardization
from tqdm import tqdm
import os


pretrained = 'casia-webface'
print(os.getcwd())
data_dir = 'dataset/'
batch_size = 32
epochs = 8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device:', device)

mtcnn = MTCNN(margin=20, keep_all=False, device=device)
dataset = datasets.ImageFolder(data_dir)
dataset.samples = [
    (p, p.replace(data_dir, data_dir+'_cropped')) for p, _ in dataset.samples
]

loader = DataLoader(dataset, batch_size=10, collate_fn=training.collate_pil)
with tqdm(total=len(loader)) as pbar:
    for (x, y) in loader:
        mtcnn(x, save_path=y)
        pbar.update(len(x))
del mtcnn

transform = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])
dataset = datasets.ImageFolder(data_dir+'_cropped', transform=transform)
img_indices = np.arange(len(dataset))
np.random.shuffle(img_indices)
train_indices = img_indices[:int(0.8 * len(img_indices))]
val_indices = img_indices[int(0.8 * len(img_indices)):]
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices))

resnet = InceptionResNetV1(classify=True, pretrained=pretrained, num_classes=len(dataset.class_to_idx)).to(device)
optimizer = optim.Adam(resnet.parameters(), lr=1e-3)
scheduler = MultiStepLR(optimizer, [5, 10])
loss_fn = torch.nn.CrossEntropyLoss()
metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}

writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10
print('\n\nInitial')
print('-' * 10)
resnet.eval()
training.pass_epoch(
    resnet, loss_fn, val_loader,
    batch_metrics=metrics, show_running=True, device=device,
    writer=writer
)

for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    resnet.train()
    training.pass_epoch(
        resnet, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )
writer.close()
