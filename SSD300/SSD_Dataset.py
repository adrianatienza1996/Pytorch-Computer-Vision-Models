import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os


IMAGE_SIZE = 300


class PascalVOC_Dataset(Dataset):
    def __init__(self, anno_df, images_path):
        self.df = anno_df
        self.images_id = np.array(pd.unique(self.df.iloc[:, 0]))
        self.path = images_path
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.images_id)

    def __getitem__(self, ix):
        image = self.images_id[ix]
        tmp = self.df.iloc[self.df.iloc[:, 0] == image]
        shape = f[1:3]
        boxes = f[3:7]
        label = f[-1]

        file = os.path.join(self.path, image_id)
        im = cv2.imread(file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im, shape, boxes, label

    def preprocess_image(self, im):
        im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
        im = torch.tensor(im).permute(2, 0, 1)
        im = self.normalize(im / 255.)
        return im[None]


    def collate_fn(self, batch):
        ims, ages, races, genders = [], [], [], []
        for im, age, race, gender in batch:
            im = self.preprocess_image(im)
            ims.append(im)
            ages.append(np.where(self.ages_values == age)[0][0])
            races.append(np.where(self.races_values == race)[0][0])
            genders.append(float(gender))

        ages, races = [torch.tensor(x).to(device).long() for x in [ages, races]]
        genders = torch.tensor(genders).to(device).float()
        ims = torch.cat(ims).to(device)
        return ims, ages, races, genders
