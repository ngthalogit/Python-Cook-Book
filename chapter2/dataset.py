from PIL import Image 
import torch  
from torch.utils.data import Dataset
from torchvision import transform
import os 
import pandas as pd
torch.manual_seed(0)

class HistoCancerDataset():
    def __init__(self, data_dir, transform, data_type='train'):
        data_path = os.path.join(data_dir, data_type)

        imgs_name = os.listdir(data_path)
        
        self.imgs_path = [os.path.join(data_path, f) for f in imgs_name]
        
        # get labels from .csv file 
        labels_csv = data_type + '_labels.csv'
        labels_csv_path = os.path.join(data_dir, labels_csv)
        labels_df = pd.read_csv(labels_csv_path)

        # set index in data frame to id 
        labels_df.set_index('id', inplace=True)

        self.labels = [labels_df.loc[img_name[:-4]].values[0] for img_name in imgs_name]

        self.trainsform = transform 

    def __len__(self):
        return len(imgs_name)

    def __getitem__(self, idx):
        img = Image.open(self.imgs_path[idx])
        img = self.transform(img) 
        return img, self.labels[idx]

    