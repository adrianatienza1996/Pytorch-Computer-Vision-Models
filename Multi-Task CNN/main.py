import torch
import torch.nn as nn
import numpy as np

from model import ResNet18_Classifier
from utils import get_data, train_batch, accuracy

import matplotlib.pyplot as plt

from torch.optim import Adam
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

gender_criterion = nn.BCELoss()
age_criterion = nn.CrossEntropyLoss()
race_criterion = nn.CrossEntropyLoss()
loss_functions = gender_criterion, race_criterion, age_criterion

model = ResNet18_Classifier().to(device)
optimizer = Adam(model.parameters (), lr= 1e-4)

epochs = 30
tr_dl, val_dl = get_data("F:/FairFace Database/fairface_label_val.csv", "F:/FairFace Database/fairface_label_val.csv", "F:/FairFace Database")

loss, accu_age, accu_race, accu_gender = [], [], [], []

for epoch in range(epochs):
    print("Epoch: ", str(epoch))
    epoch_loss, epoch_accu_age, epoch_accu_race, epoch_accu_gender = [], [], [], 0.0

    for ix, batch in enumerate(iter(tr_dl)):
        batch_loss = train_batch(batch, model, optimizer, loss_functions)
        epoch_loss.append(batch_loss)

    epoch_loss = np.array(epoch_loss).mean()

    for ix, batch in enumerate(iter(val_dl)):
        is_correct_age, is_correct_race, accu_gender_batch = accuracy(batch, model)
        epoch_accu_age.extend(is_correct_age)
        epoch_accu_race.extend(is_correct_race)

    accu_gender_batch = accu_gender_batch / 10954.0
    epoch_accu_age = np.mean(epoch_accu_age)
    epoch_accu_race = np.mean(epoch_accu_race)

    loss.append(epoch_loss)
    accu_age.append(epoch_accu_age)
    accu_gender.append(accu_gender_batch)
    accu_race.append(epoch_accu_race)

    print("Epoch loss: " + str(epoch_loss))
    print("Age accuracy: " + str(epoch_accu_age))
    print("Accu gender: " + str(accu_gender_batch))
    print("Race accuracy: " + str(epoch_accu_race))

    torch.save(model.to("cpu").state_dict(), "Saved_Model/my_model.pth")
    print("Model Saved")
    model.to(device)

plt.plot(range(epochs), epoch_loss)
plt.plot(range(epochs), epoch_accu_age)
plt.plot(range(epochs), accu_gender)
plt.plot(range(epochs), epoch_accu_race)

plt.show()
