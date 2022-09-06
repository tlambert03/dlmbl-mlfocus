import os

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler

import wandb
from mlfocus.loader import MIPData
from mlfocus.model import ResnetModel
from mlfocus.train import train_step, validate

os.environ["WANDB_NOTEBOOK_NAME"] = "mip_train.ipynb"
wandb.init(project="mip-mlfocus")



full_dset = MIPData("../mlfocus_data/")


model = ResnetModel(in_channels=1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
batch_size = 1024
patches_per_image = 20
learning_rate = 1.5e-3
epochs = 200

train_dataset = MIPData("../mlfocus_data/", exclude=["_data/beads"])
sampler = RandomSampler(
    train_dataset, replacement=True, num_samples=len(train_dataset) * patches_per_image
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=4,
    pin_memory=True,
)
# mt_data = MIPData("../mlfocus_data/", include=["mt003"])
# mt_loader = DataLoader(mt_data, batch_size=64)
bead_data = MIPData("../mlfocus_data/", include=["_data/beads"])
bead_loader = DataLoader(bead_data, batch_size=64)
all_loader = DataLoader(full_dset, batch_size=64)


optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

wandb.config = {
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": batch_size,
    "patches_per_image": patches_per_image,
}
wandb.watch(model)

global_len = len(train_loader.dataset)
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
    # mt_val = validate(model, mt_loader, criterion)
    bead_val = validate(model, bead_loader, criterion)
    wandb.log(
        {"tot_val_loss": tot_val, "bead_val": bead_val, "epoch": e}
        # {"tot_val_loss": tot_val, "mt_val": mt_val, "bead_val": bead_val, "epoch": e}
    )

torch.save(model, 'models/mip_model_nobeads.pt')
