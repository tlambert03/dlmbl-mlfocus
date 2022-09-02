from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.cuda
from torch import Tensor, argmax, sum, mean
import toolz as tz


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(
    loader: DataLoader, model: nn.Module, optimizer: optim.Optimizer, loss: nn.Module
) -> Tensor:
    """Train the model for one epoch."""

    # set the model into train mode
    model.train()

    losses = Tensor(0)
    for x, y in tqdm(loader, "train"):

        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        y_pred = model(x)
        l: Tensor = loss(y_pred, y)
        l.backward()

        optimizer.step()
        losses.add(l.item())

    return mean(losses)


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


validate = evaluate(name="validate")
test = evaluate(name="test")
