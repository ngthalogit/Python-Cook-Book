from torchvision import transforms
from dataset import HistoCancerDataset
from torch.utils.data import random_split, DataLoader
from model import Net
from torchsummary import summary
import torch
from torch.optim.lr_schduler import ReduceLROnPlateau

# convert PIL images to Tensor image 
tensor_transform = transforms.Compose([transforms.ToTensor()])

data_dir = ''
dataset = HistoCancerDataset(data_dir, tensor_transform)
len_dataset = len(dataset)

# split dataset into training and validation set 
len_train = int(0.8 * len(dataset))
len_val = len_dataset - len_train

train_ds, val_ds = random_split(dataset, [train_ds, val_ds])

"""
SHOW IMAGE 

import numpy as np 
import matplotlib.pyplot as plt

def show(img, y, color=False):
    npimg = y.numpy()
    npimg_tr = np.transpose(npimg, (1, 2, 0))
    if color: 
        plt.imshow(npimg_tr, interpolation='nearest')
    else: 
        plt.imshow(npimg_tr[:, :, 0], interpolation='nearest', cmap='gray')

"""

# augmentation for training set 
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(45),
    transforms.RandomResizedCrop(96, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
    transforms.ToTensor()
])

train_ds.transforms = data_transform
val_ds.transforms = tensor_transform

# create DataLoader 
train_dl = Dataloader(train_ds, batch_size=32, shuffle=True) 
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

# get model
model = Net()
device = torch.device('cuda:0')
model = model.to(device)
print(model)
print(summary(model, input_size=(3, 96, 96), deivce=device.type))

# define loss function 
loss_func = nn.NLLLoss(reduction='sum')
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
opt.zero_grad()

# define learning rate scheduler
lr_schduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20, verbose=1)

# training   

for epoch in range(100):
    model.train()
    model.eval()
    with torch.no_grad():
        pass
