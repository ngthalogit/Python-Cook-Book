from model import Net
from matplotlib import pyplot as plt
from torchvision import datasets
import torch


data_save_path = '/content'
load_path = '/content/weight.pt'

# load model
model = Net()
model.load_state_dict(torch.load(load_path))

# test from validation
downloaded = True if os.path.exists(data_save_path) else False
val_data = datasets.MNIST(data_save_path, train=False, download=downloaded)

images, labels = val_data.data, val_data.targets

# get image
test_img = images[20]
test_img = test_img.unsqueeze(0)
test_label = labels[20]
# show image
plt.imshow(test_img.numpy()[0], cmap='gray')

test_img = test_img.unsqueeze(0)  # [1, 1, 28, 28]
test_img = test_img.type(torch.float)


output = model(test_img)
pred = output.argmax(dim=1, keepdims=True)

print('Labels: %d, Prediction: %d' % (test_label.item(), pred.item()))
