from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
from model import Net
import torch 
import os

# save on google colab
data_save_path = '/content' 

# define transform 
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomVerticalFlip(p=1),
    transforms.ToTensor()
])

downloaded = True if os.path.exists(data_save_path) else False
# loading  training and validation data  
train_data = datasets.MNIST(data_save_path, train=True, download=downloaded, transform=data_transform)
val_data = datasets.MNIST(data_save_path, train=False, download=downloaded)

# extract data and targets 
x_train, y_train = train_data.data, train_data.targets # [60000, 28, 28], [60000]
x_val, y_val = val_data.data, val_data.targets # [10000, 28, 28], [10000]

# convert B * H * W to B * C * H * W
x_train = x_train.unsqueeze(1) # [60000, 1, 28, 28]
x_val = x_val.unsqueeze(1) # [10000, 1, 28, 28]

# wrapping tensors into dataset 
train_ds = TensorDataset(x_train, y_train) 
val_ds = TensorDataset(x_val, y_val)

# create dataloader 
train_dl = DataLoader(train_ds, batch_size=8)
val_dl = DataLoader(val_ds, batch_size=8)

# get model
model = Net()
device = torch.device('cuda:0')
model.to(device)

# define loss function and optimizer
loss_func = torch.nn.NLLLoss(reduction='sum')
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
opt.zero_grad()

# training
def loss_batch(loss_func, x_batch, y_batch, y_hat, optimizer=None):
    loss = loss_func(y_hat, y_batch)
    pred = y_hat.argmax(dim=1, keepdim=True)
    corrects = pred.eq(y_batch.view_as(pred)).sum().item()
    if optimizer is not None: 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.item(), corrects

for epoch in range(5):
    # training epoch 
    model.train()
    loss_train, metric_train = 0.0, 0.0 
    len_train_dl = len(train_dl.dataset)
    # training batch 
    for x_batch, y_batch in train_dl:
        x_batch = x_batch.type(torch.float).to(device)
        y_batch = y_batch.to(device)

        y_hat = model(x_batch)

        loss_b, metric_b = loss_batch(loss_func, x_batch, y_batch, y_hat, opt)
        loss_train += loss_b 
        if metric_b is not None:
            metric_train += metric_b
    loss_train /= len_train_dl
    metric_train /= len_train_dl
    model.eval()
    with torch.no_grad():
        loss_val, metric_val = 0.0, 0.0 
        len_val_dl = len(val_dl.dataset)
        for x_batch, y_batch in val_dl:
            x_batch = x_batch.type(torch.float).to(device)
            y_batch = y_batch.to(device)
            
            y_hat = model(x_batch)
            loss_b, metric_b = loss_batch(loss_func, x_batch, y_batch, y_hat, opt)
            loss_val += loss_b 
            if metric_b is not None:
                metric_val += metric_b 
        loss_val /= len_val_dl
        metric_val /= len_val_dl

    accuracy = 100 * metric_val
    print('epoch: %d, train_loss: %.6f, val_loss: %.6f, accuracy: %.2f' %(epoch, loss_train, loss_val, accuracy))

# saving model 
model_save_path = '/content/weight.pt'
torch.save(model.state_dict(), model_save_path)







