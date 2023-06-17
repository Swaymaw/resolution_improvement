import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch


class MapDataset(Dataset):
    def __init__(self):
        self.hr_image_files = sorted(os.listdir('data/hr_images'))
        self.lr_image_files = sorted(os.listdir('data/lr_images'))

    def __len__(self):
        return len(self.hr_image_files)

    def __getitem__(self, index):
        hr_img_file = self.hr_image_files[index]
        hr_img_path = os.path.join('data/hr_images', hr_img_file)
        lr_img_path = os.path.join('data/lr_images', hr_img_file)

        hr_image = np.array(Image.open(hr_img_path)).astype('float32')
        lr_image = np.array(Image.open(lr_img_path)).astype('float32')        
        input_image = torch.from_numpy(lr_image)
        target_image =torch.from_numpy(hr_image)

        return input_image.to(config.DEVICE), target_image.to(config.DEVICE)
