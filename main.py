from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from model import Model, ResNet
import re
import pickle
import numpy as np
import pandas as pd
import warnings
import torch.nn as nn
import json
import torch.nn.functional as F
from time import time
warnings.filterwarnings("ignore")

batch_size = 64
num_image = 60000
epochs = 90

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
data_root_path = "data/"
train_dataset = datasets.FashionMNIST(root=data_root_path, train=True,
                                      transform=train_transform, download=True)
test_dataset = datasets.FashionMNIST(root=data_root_path, train=False,
                                     transform=test_transform, download=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True,
                          num_workers=4,
                          )
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=True,
                         )


# model = ResNet(1, 28, 10)
model = Model(1, 28, 10)
if torch.cuda.is_available():
    model.cuda()
# optimizer = torch.optim.SGD(
#     model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, epochs, eta_min=0.001)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
opt_accu = -1
for i in range(epochs):
    model.train()
    loss_sum = 0
    # lr_scheduler.step()
    st_time = time()
    for imgs, label in train_loader:
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            label = label.cuda()
        preds = model(imgs)
        loss = F.cross_entropy(preds, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(imgs)/num_image
    model.eval()
    ncorrect = 0
    nsample = 0
    valloss = 0
    for imgs, label in test_loader:
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            label = label.cuda()
        preds = model(imgs)
        ncorrect += torch.sum(preds.max(1)[1].eq(label).double())
        nsample += len(label)
        loss = F.cross_entropy(preds, label)
        valloss += loss.item() * len(imgs)
    val_accu, val_loss = float((ncorrect/nsample).cpu()), valloss/nsample
    if val_accu > opt_accu:
        opt_accu = val_accu
    print(f"Epoch~{i+1}->train_loss:{round(loss_sum,4)}, val_loss:{round(val_loss, 4)}, val_accu:{round(val_accu, 4)}, time:{round(time()-st_time,4)}")

torch.save(model.state_dict(), model.tag)
