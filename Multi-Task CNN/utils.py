import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import pandas as pd
import numpy as np
import os
device = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 224

class GenderAgeDataset(Dataset):
    def __init__(self, df, path):
        self.df = df
        self.path = path
        self.ages_values = np.array(pd.unique(self.df.loc[:, "age"]).squeeze())
        self.races_values = np.array(pd.unique(self.df.loc[:, "race"]).squeeze())
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        f = self.df.iloc[ix].squeeze()
        file = f.file
        file = os.path.join(self.path, file)
        gen = f.gender == 'Female'
        age = f.age
        race = f.race
        im = cv2.imread(file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im, age, race, gen

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

def get_data (train_csv_path, test_csv_path, path):
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    trn = GenderAgeDataset(train_df, path)
    val = GenderAgeDataset(test_df, path)

    train_loader = DataLoader(trn, batch_size=32, shuffle=True, drop_last=True, collate_fn=trn.collate_fn)
    test_loader = DataLoader(val, batch_size=test_csv_path.shape[0],  collate_fn=val.collate_fn)
    return train_loader, test_loader


def train_batch(data, model, opt, loss_fn):
    model.train()
    gender_criterion, race_criterion, age_criterion = loss_fn
    img, age, race, gender = data
    prediction_age, prediction_race, prediction_gender = model(img)
    l2_regularization = 0
    for param in model.parameters():
        l2_regularization += torch.norm(param, 2)

    age_loss = age_criterion(prediction_age.squeeze(), age)
    race_loss = race_criterion(prediction_race.squeeze(), race)
    gender_loss = gender_criterion(prediction_gender.squeeze(), gender)

    batch_loss = age_loss + race_loss+ gender_loss + 0.01 * l2_regularization
    batch_loss.backward()
    opt.step()
    opt.zero_grad()
    return batch_loss.item()

@torch.no_grad()
def accuracy(data, model):
    model.eval()
    img, age, race, gender = data
    prediction_age, prediction_race, prediction_gender = model(img)
    max_values_age, argmaxes_age = prediction_age.max(-1)
    is_correct_age = argmaxes_age == age

    max_values_race, argmaxes_race = prediction_race.max(-1)
    is_correct_race = argmaxes_race == race

    prediction_gender = (prediction_gender > 0.5).squeeze()
    gender_account = (prediction_gender == gender).float().cpu().numpy().sum()

    return is_correct_age.cpu().numpy().tolist(), is_correct_race.cpu().numpy().tolist(), gender_account








