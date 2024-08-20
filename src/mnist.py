"""Identify MNIST digits."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torchvision.datasets as tvd
import torchvision.transforms as tvt
from tqdm import tqdm


class Net(th.nn.Module):
    """MNIST Network."""

    def __init__(self) -> None:
        """Network initialization.

        Use Lenet5 as inspiration.
        https://en.wikipedia.org/wiki/LeNet#/media/File:LeNet_architecture.png
        """
        super().__init__()
        self.layer1 = th.nn.Conv2d(1, 16, 3)
        self.pool1 = th.nn.MaxPool2d(2, stride=2)
        self.layer2 = th.nn.Conv2d(16, 32, 3)
        self.pool2 = th.nn.MaxPool2d(2, stride=2)
        self.layer3 = th.nn.Conv2d(32, 64, 3)
        self.layer4 = th.nn.Linear(576, 128)
        self.layer5 = th.nn.Linear(128, 10)
        self.relu = th.nn.ReLU()
        self.sigmoid = th.nn.Sigmoid()

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Network forward pass.

        Args:
            x (th.Tensor): Input Tensor of shape (BS, 1, 28, 28).

        Returns:
            th.Tensor: Network predictions of shape (BS, 10).
        """
        x = self.relu(self.layer1(x))
        x = self.pool1(x)
        x = self.relu(self.layer2(x))
        x = self.pool2(x)
        x = self.relu(self.layer3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.layer4(x)
        x = self.layer5(x)
        return self.sigmoid(x)


def cross_entropy(label: th.Tensor, out: th.Tensor) -> th.Tensor:
    """Cross Entropy loss.

    Args:
        label (th.Tensor): Ground truth labels.
        out (th.Tensor): Network predictions.

    Returns:
        th.Tensor: Cross-Entropy loss.
    """
    left = label * th.log(out + 1e-12)
    right = (1 - label) * th.log(1 - out + 1e-12)
    sum_ = th.sum(left + right)
    return -sum_ / np.prod(label.size())


def sgd_step(model: Net, learning_rate: float) -> Net:
    """Perform SGD.

    Args:
        model (Net): Network objekt.
        learning_rate (float): Learning rate or step size.

    Returns:
        Net: SGD applied model.
    """
    for param in model.parameters():
        param.data = param.data - learning_rate * param.grad.data
    return model


def get_acc(model: Net, dataloader: th.utils.data.DataLoader) -> float:
    """Compute accuracy given specific dataloader.

    Args:
        model (Net): Network objekt.
        dataloader (th.utils.data.DataLoader): Dataloader objekt.

    Returns:
        float: Accuracy.
    """
    acc = []
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs)
        rec = th.argmax(preds, dim=1)
        batch_acc = th.sum((rec == labels).type(th.float32)) / len(labels)
        acc.append(batch_acc.cpu())
    return float(sum(acc) / len(acc))


def zero_grad(model: Net) -> Net:
    """Make gradients zero after SGD.

    Args:
        model (Net): Network object.

    Returns:
        Net: Network with zeroed gradients.
    """
    for param in model.parameters():
        param.grad.data.zero_()
    return model


# HYPERPARAMETERS
BS = 200
EPOCHS = 10
DEVICE = th.device("cuda") if th.cuda.is_available() else th.device("cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train network on MNIST.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning Rate")
    args = parser.parse_args()
    print(args)

    train_transforms = tvt.Compose(
        [
            tvt.ToTensor(),
            tvt.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )
    test_transforms = tvt.Compose(
        [
            tvt.ToTensor(),
            tvt.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )
    dataset = tvd.MNIST(
        "./.data", train=True, download=True, transform=train_transforms
    )
    trainset, valset = th.utils.data.random_split(dataset, [50000, 10000])

    train_loader = th.utils.data.DataLoader(trainset, batch_size=BS, shuffle=True)
    val_loader = th.utils.data.DataLoader(valset, batch_size=BS, shuffle=False)
    test_loader = th.utils.data.DataLoader(
        tvd.MNIST("./.data", train=False, download=True, transform=test_transforms),
        batch_size=10000,
        shuffle=False,
    )

    train_accs = []
    val_accs = []
    test_accs = []
    for seed in range(5):
        print(f"Seed: {seed}")
        th.manual_seed(seed)
        model = Net()
        model = model.to(DEVICE)
        per_epoch_train_acc = []
        per_epoch_val_acc = []
        for e in range(EPOCHS):
            epoch_loss = []
            print(f"Epoch: {e}")
            for imgs, labels in tqdm(train_loader):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

                preds = model(imgs)
                loss_val = cross_entropy(
                    label=th.nn.functional.one_hot(labels, num_classes=10), out=preds
                )
                loss_val.backward()

                model = sgd_step(model, learning_rate=args.lr)
                model = zero_grad(model)
                epoch_loss.append(loss_val.item())
            print(f"Loss: {sum(epoch_loss)/len(epoch_loss):2.4f}")

            train_acc = get_acc(model=model, dataloader=train_loader)
            val_acc = get_acc(model=model, dataloader=val_loader)
            per_epoch_train_acc.append(train_acc)
            per_epoch_val_acc.append(val_acc)
            print(f"train acc: {train_acc:2.4f}, val acc: {val_acc:2.4f}")
        test_acc = get_acc(model=model, dataloader=test_loader)
        train_accs.append(per_epoch_train_acc)
        val_accs.append(per_epoch_val_acc)
        test_accs.append(train_acc)
    train_accs_np = np.stack(train_accs, axis=0)
    val_accs_np = np.stack(val_accs, axis=0)
    test_accs_np = np.stack(test_accs)

    train_mean = np.mean(train_accs_np, axis=0)
    val_mean = np.mean(val_accs_np, axis=0)
    test_mean = np.mean(test_accs_np)

    train_std = np.std(train_accs_np, axis=0)
    val_std = np.std(val_accs_np, axis=0)
    test_std = np.std(test_accs_np)

    def plot_mean_std(steps, mean, std, color, label="", marker="."):
        """Plot means and standard deviations with shaded areas."""
        plt.plot(steps, mean, label=label, color=color, marker=marker)
        plt.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_mean_std(
        np.arange(1, 1 + EPOCHS), train_mean, train_std, colors[0], label="train acc"
    )
    plot_mean_std(
        np.arange(1, 1 + EPOCHS), val_mean, val_std, colors[1], label="val acc"
    )
    plt.errorbar(
        np.array([EPOCHS]),
        test_mean,
        test_std,
        color=colors[2],
        label="test acc",
        marker="x",
    )
    plt.legend()
    plt.savefig("./figures/acc.png")
    print("done")
