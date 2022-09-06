import sys

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import wandb
from mlfocus.loader import MyFFTData
from mlfocus.localmodel import SuperModel
from mlfocus.train import train_step, validate


full_dset = MyFFTData("../mlfocus_data/")


model = SuperModel()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
batch_size = 512
learning_rate = 1e-3
epochs = 200

train_dataset = MyFFTData("../mlfocus_data/", exclude=["_data/beads3"])
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
val_data = MyFFTData("../mlfocus_data/", include=["_data/beads3"])
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
all_loader = DataLoader(full_dset, batch_size=batch_size, shuffle=True)

optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

train_step(
    train_loader,
    model=model,
    optimizer=optimizer,
    loss=criterion,
    device=device,
    sanity_check=True,
)


wandb.init(project="fft-local-mlfocus")
wandb.watch(model)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for e in range(epochs):
    train_step(
        train_loader,
        model=model,
        optimizer=optimizer,
        loss=criterion,
        epoch_idx=e,
        device=device,
    )
    tot_val = validate(model, all_loader, criterion)
    loss_val = validate(model, val_loader, criterion)
    # wandb.log({"tot_val_loss": tot_val, "mt_val_loss": mt_val, "epoch": e})
    wandb.log({"tot_val_loss": tot_val, "bead_val": loss_val, "epoch": e})

from datetime import datetime

torch.save(model, f"models/fft_model_{datetime.now()}.pt")
