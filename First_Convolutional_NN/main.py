import torch
from torch import optim
#from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

from utils import get_data, get_model, train_batch, val_loss, accuracy


from torchvision import datasets
data_folder = '~/Data'

fmnist = datasets.FashionMNIST(data_folder, download=True, train=True)
tr_images = fmnist.data
tr_targets = fmnist.targets
print ("Data Shape: " + str(tr_images.shape))
print ("Target Shape: " + str(tr_targets.shape))

val_fmnist = datasets.FashionMNIST(data_folder,download=True, train=False)
val_images = val_fmnist.data
val_targets = val_fmnist.targets
print ("Data Shape: " + str(val_images.shape))
print ("Target Shape: " + str(val_targets.shape))

# Training procedure

epochs = 5
model, loss_fn, optimizer = get_model()
#summary(model, (1, 28, 28))

tr_dl, val_dl = get_data(tr_images, tr_targets, val_images, val_targets)

scheduler = optim.lr_scheduler.ReduceLROnPlateau (optimizer,
                                                 factor = 0.5,
                                                 patience = 0,
                                                 threshold = 0.001,
                                                 verbose = True,
                                                 min_lr = 1e-5,
                                                 threshold_mode = "abs")


loss, accu = [], []

for epoch in range (epochs):
    print ("Epoch: ", str(epoch))
    epoch_loss, epoch_accu = [], []
    
    for ix, batch in enumerate(iter(tr_dl)):
        x, y = batch
        batch_loss = train_batch(x, y, model, optimizer, loss_fn)
        epoch_loss.append(batch_loss)
    
    epoch_loss = np.array(epoch_loss).mean()
    
    for ix, batch in enumerate(iter(val_dl)):
        x, y = batch
        is_correct = accuracy(x, y, model)
        epoch_accu.extend(is_correct)
    
    epoch_accu = np.mean(epoch_accu)
    
    for ix, batch in enumerate(iter(val_dl)):
        x, y = batch
        validation_loss = val_loss (x, y, model, loss_fn)
        scheduler.step(validation_loss)
        
    loss.append(epoch_loss)
    accu.append(epoch_accu)

torch.save(model.to("cpu").state_dict(), "Saved_Model/my_model.pth")
print ("Model Saved")

