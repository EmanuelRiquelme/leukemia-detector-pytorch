import os
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import pandas as pd

class train_cancer(Dataset):
    def __init__(self, root_dir = 'C-NMC_training_data',transform = None):
      self.root_dir = root_dir
      self.transform = transform if transform else transforms.Compose([
                    transforms.CenterCrop(224),
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                    ])

    def __files_name__(self):
        dirs = ['fold_1','fold_2','fold_0']
        sub_dirs = ['all','hem']
        file_names = []
        for dir in dirs:
            for sub_dir in sub_dirs:
                file_names.extend([f'{self.root_dir}/{dir}/{sub_dir}/{file}' for file in os.listdir(f'{self.root_dir}/{dir}/{sub_dir}')])
        return file_names

    def __len__(self):
        return len(self.__files_name__())

    def __getitem__(self, idx):
        file_name = self.__files_name__()[idx]
        label = file_name.split('/')[-1].split('_')[4].split('.')[0]
        label = torch.tensor(1) if label == 'all' else torch.tensor(0)
        img = self.transform(Image.open(file_name))
        return img,label

class test_cancer(Dataset):
    def __init__(self, root_dir = 'C-NMC_test_prelim_phase_data',transform = None):
      self.root_dir = root_dir
      self.df = pd.read_csv(f'{self.root_dir}/{self.root_dir}_labels.csv')
      self.transform = transform if transform else transforms.Compose([
                    transforms.CenterCrop(224),
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                    ])
    def __files_name__(self):
        return [file_name for file_name in os.listdir(f'{self.root_dir}/{self.root_dir}')]

    def __len__(self):
        return len(self.__files_name__())
    def __getitem__(self,idx):
        file_name = self.__files_name__()[idx]
        label =  self.df[self.df['new_names'] == file_name]['labels'].to_numpy()
        img = Image.open(f'{self.root_dir}/{self.root_dir}/{file_name}')
        return self.transform(img),torch.tensor(label)
