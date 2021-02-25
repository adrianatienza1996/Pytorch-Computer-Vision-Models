import torchsummary
import torch
from model import MultiTask_EfficientNet
from utils import get_data, train_batch, accuracy

from torch.optim import Adam
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = MultiTask_EfficientNet("b0").to(device)
print(type(model))
print("Model Loaded")

epochs = 5
tr_dl, val_dl = get_data("F:/FairFace Database/fairface_label_train.csv", "F:/FairFace Database/fairface_label_val.csv", "F:/FairFace Database")

loss, accu_age, accu_race, accu_gender = [], [], [], []

for epoch in range(epochs):
    print("Epoch: ", str(epoch))
    epoch_loss, epoch_accu_age, epoch_accu_race, accu_gender = [], [], [], 0.0

    for ix, batch in enumerate(iter(tr_dl)):
        batch_loss = train_batch(batch, model, optimizer, loss_functions)
        epoch_loss.append(batch_loss)

    epoch_loss = np.array(epoch_loss).mean()

    for ix, batch in enumerate(iter(val_dl)):
        is_correct_age, is_correct_race, accu_gender = accuracy(batch, model)
        epoch_accu_age.extend(is_correct_age)
        epoch_accu_race.extend(is_correct_race)


    epoch_accu_age = np.mean(epoch_accu_age)
    epoch_accu_race = np.mean(epoch_accu_race)
    accu_gender = accu_gender / 10954.0

    loss.append(epoch_loss)
    accu_age.append(epoch_accu_age)
    accu_gender.append(accu_gender)
    accu_race.append(epoch_accu_race)

torch.save(model.to("cpu").state_dict(), "Saved_Model/my_model.pth")
print("Model Saved")
