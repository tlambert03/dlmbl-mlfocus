from typing import List, Optional
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.cuda
from torch import Tensor, argmax, sum, mean
import toolz as tz
import wandb
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_step(
    loader: DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss: nn.Module,
    *,
    log_interval: int = 100,
    epoch_idx: Optional[int] = None,
    device=DEVICE,
    sanity_check: bool = False,
) -> List[float]:
    """Train the model for one epoch."""

    # set the model into train mode
    model.train()

    losses = []
    nbatches = len(loader)
    nitems = len(loader.dataset)

    pbar = tqdm(enumerate(loader), "train", total=nbatches)

    for batch_idx, (x, y) in pbar:

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        y_pred = model(x)
        l: Tensor = loss(y_pred, y)
        l.backward()

        optimizer.step()
        losses.append(l.item())
        
        if sanity_check:
            print("sanity check", l.item())
            return losses
        
        wandb.log({"loss": float(l.item()), "epoch": epoch_idx})

        if log_interval and batch_idx % log_interval == 0:
            pbar.set_description(
                f"Epoch: {epoch_idx} [{batch_idx * len(x)}/{nitems}]\tLoss: {l.item():.6f}"
            )

    return losses


@tz.curry
def evaluate(data: Dataset, model: nn.Module, name: str, batch_size: int = 32):
    model.eval()

    loader = DataLoader(data, batch_size=batch_size)
    correct = 0
    total = 0
    for x, y in tqdm(loader, name):
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits = model(x)
        probs = nn.Softmax(dim=1)(logits)
        predictions = argmax(probs, dim=1)

        correct += int(sum(predictions == y).cpu().detach().numpy())
        total += len(y)

    return correct / total


# validate = evaluate(name="validate")
# test = evaluate(name="test")


# run validation after training epoch
def validate(model, loader, criterion, device=DEVICE):
    # set model to eval mode
    model.eval()
    model.to(device)

    # running loss and metric values
    tot_val_loss = 0

    # disable gradients during validation
    with torch.no_grad():
        # iterate over validation loader and update loss and metric values
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            tot_val_loss += criterion(prediction, y).item()

    # normalize loss and metric
    tot_val_loss /= len(loader)

    # log additional val losses here

    return tot_val_loss
