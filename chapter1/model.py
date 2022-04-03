import torch.nn as nn 
import torch.nn.functional as F 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=20, 
            kernel_size=5,
        )
        self.conv2 = nn.Conv2d(
            in_channels=20, 
            out_channels=50,
            kernel_size=5
        )
        self.fc1 = nn.Linear(in_channels=4*4*50, out_channels=500)
        self.fc2 = nn.Linear(in_channels=500, out_channels=10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, in_channels=2, out_channels=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2d(x, in_channels=2, out_channels=2)
        x = x.view(-1, 4*4*50)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x