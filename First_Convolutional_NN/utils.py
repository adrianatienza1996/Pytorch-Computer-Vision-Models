from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.optim import Adam
device = "cuda" if torch.cuda.is_available() else "cpu"


class FMNISTDataset(Dataset):
    def __init__(self, x, y, aug=None):
        self.x, self.y = x, y
        self.aug = aug

    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        return x, y

    def __len__(self): return len(self.x)

    def collate_fn(self, batch):
        'logic to modify a batch of images'
        ims, classes = list(zip(*batch))
        if self.aug: ims=self.aug.augment_images(images=ims)

        ims = torch.tensor(ims)[:,None,:,:].to(device)/255.
        classes = torch.tensor(classes).to(device)
        return ims, classes


def get_data(tr_images, tr_targets, val_images, val_targets, aug):
    train = FMNISTDataset(tr_images, tr_targets, aug=aug)
    trn_dl = DataLoader(train, batch_size=64, collate_fn=train.collate_fn, shuffle=True)
    val = FMNISTDataset(val_images, val_targets)
    val_dl = DataLoader(val, batch_size=len(val_images), collate_fn=val.collate_fn, shuffle=True)
    return trn_dl, val_dl

def get_model():
    
    model = nn.Sequential(
        nn.Conv2d(1 , 64, kernel_size = 3, padding = True),
        nn.MaxPool2d(2),
        nn.ReLU(),
        
        nn.Conv2d(64 , 128, kernel_size = 3, padding = True),
        nn.MaxPool2d(2),
        nn.ReLU(),
        
        nn.Flatten(),
        nn.Linear(7 * 7 * 128, 256),
        nn.ReLU(),
        nn.Linear(256, 10)).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-2)
    
    return model, loss_fn, optimizer
    
def train_batch (x, y, model, opt, loss_fn):
    model.train()
    prediction = model(x)
    l2_regularization = 0
    
    for param in model.parameters():
        l2_regularization += torch.norm(param, 2)
        
    batch_loss = loss_fn(prediction, y) + 0.01 * l2_regularization
    batch_loss.backward()
    opt.step()
    opt.zero_grad()
    return batch_loss.item()


@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    max_values, argmaxes = prediction.max(-1)
    is_correct = argmaxes == y
    return is_correct.cpu().numpy().tolist()

@torch.no_grad()
def val_loss(x, y, model, loss_fn):
    model.eval()
    prediction = model(x)
    val_loss = loss_fn(prediction, y)
    return val_loss.item()