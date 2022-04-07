from torchvision import transforms
from dataset import HistoCancerDataset
from torch.utils.data import random_split, DataLoader
from model import Net
from torchsummary import summary
import torch
from torch.optim.lr_schduler import ReduceLROnPlateau
from untils import loss_batch
import copy

# convert PIL images to Tensor image
tensor_transform = transforms.Compose([transforms.ToTensor()])

data_dir = ''
dataset = HistoCancerDataset(data_dir, tensor_transform)
len_dataset = len(dataset)

# split dataset into training and validation set
len_train = int(0.8 * len(dataset))
len_val = len_dataset - len_train

train_ds, val_ds = random_split(dataset, [len_train, len_val])

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
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

# get model
model = Net()
device = torch.device('cuda:0')
model = model.to(device)
print(model)
print(summary(model, input_size=(3, 96, 96), device=device.type))

# define loss function
loss_func = nn.NLLLoss(reduction='sum')
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
opt.zero_grad()

# define learning rate scheduler
lr_schduler = ReduceLROnPlateau(
    opt, mode='min', factor=0.5, patience=20, verbose=1)

# training
sanity = True
loss_hist = {
    'train': [], 
    'val': []
}
metric_hist = {
    'train': [],
    'val': []
}
best_model = copy.deepcopy(model.state_dict())
best_loss = float('inf')
weights_save_path = ''
for epoch in range(100):
    curr_lr = get_lr(opt)
    print('Epoch {}/{}, current lr={}'.format(epoch, 99, curr_lr))
    model.train()
    loss_train, metric_train = 0.0, 0.0
    len_train_dl = len(train_dl.dataset)
    for x_batch, y_batch in train_dl:
        x_batch = x_batch.type(torch.float).to(device)
        y_batch = y_batch.to(device)
        y_hat = model(x_batch)
        loss_b, metric_b = loss_batch(loss_func, y_batch, y_hat, opt)
        loss_train += loss_b
        if metric_b is not None:
            metric_train += metric_b
        loss_hist['train'].append(loss_b)
        metric_hist['train'].append(metric_b)
    loss_train /= float(len(train_dl.dataset))
    metric_train /= float(len(train_dl.dataset))
    if not sanity:
        break
    model.eval()
    with torch.no_grad():
        loss_val, metric_val = 0.0, 0.0
        len_val_dl = len(val_dl.dataset)
        for x_batch, y_batch in val_dl:
            x_batch = x_batch.type(torch.float).to(device)
            y_batch = y_batch.to(device)
            y_hat = model(x_batch)
            loss_b, metric_b = loss_batch(loss_func, y_batch, y_hat, opt)
            loss_train += loss_b 
            if metric_b is not None:
                metric_train += metric_b
            loss_hist['val'].append(loss_b)
            metric_hist['val'].append(metric_b)
        loss_val /= float(len(val_dl.dataset))
        metric_val /= float(len(val_dl.dataset))

    if loss_val < best_loss:
        best_loss = loss_val
        best_model = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), weights_save_path)
        print('copied best model')
    
    lr_schduler.step(val_loss)
    if curr_lr != get_lr(opt):
        print('loading best model')
        model.load_state_dict(best_model)
    print('train_loss: %.6f, val_loss: %.6f, accuracy: %.2f' %(loss_val, loss_val, 100 * metric_val))

