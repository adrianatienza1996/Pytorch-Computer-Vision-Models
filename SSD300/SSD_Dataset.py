import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
device = "cuda" if torch.cuda.is_available() else "cpu"


class PascalVOC_Dataset(Dataset):
    def __init__(self, anno_df, images_path, w=300, h=300):
        self.df = np.array(anno_df)
        self.images_id = np.array(pd.unique(self.df.iloc[:, 0]))
        self.path = images_path

        self.w = w
        self.h = h

        self.label2target = {l:t+1 for t, l in enumerate(self.df[:, -1])}
        self.label2target["background"] = 0

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.images_id)

    def __getitem__(self, ix):
        image = self.images_id[ix]
        tmp = self.df.iloc[self.df.iloc[:, 0] == image]
        boxes = tmp[3:7]
        label = tmp[-1]

        """
        boxes[:, [0, 2]] *= self.w
        boxes[:, [1, 3]] *= self.h
        """

        img_path = os.path.join(self.path, image)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img.resize((self.w, self.h), resample=Image.BILINEAR)) / 255.
        return img, boxes, label

    def collate_fn(self, batch):
        images, boxes, labels = [], [], []
        for img, boxes_img, labels_img in batch:
            img = self._preprocess_image(img)
            images.append(img)
            boxes.append(torch.tensor(boxes_img).float().to(device))
            labels.append(torch.tensor([self.label2target[c] for c in labels_img]).long().to(device))

        images = torch.cat(images).to(device)
        return images, boxes, labels

    def _preprocess_image(self, im):
        im = torch.tensor(im).permute(2, 0, 1)
        im = self.normalize(im)
        return im[None]


