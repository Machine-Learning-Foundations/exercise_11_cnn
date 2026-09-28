"""Test the python functions from src/mnist."""

import sys

import numpy as np
import torch as th

sys.path.insert(0, "./src/")

from src.mnist import Net, cross_entropy, get_acc, sgd_step, zero_grad


def test_cross_entropy() -> None:
    """Test if the cross entropy is implemented correctly (soft labels)."""
    th.manual_seed(0)
    label = th.nn.functional.sigmoid(th.randn([200, 10]))
    out = th.nn.functional.sigmoid(th.randn([200, 10]))
    result = cross_entropy(label=label, out=out)
    true_ce = th.nn.functional.binary_cross_entropy(input=out, target=label)
    assert np.allclose(result, true_ce)


def test_cross_entropy_one_hot() -> None:
    """Test the cross entropy with integer one-hot labels as used in training."""
    th.manual_seed(0)
    label = th.nn.functional.one_hot(th.randint(0, 10, (200,)), num_classes=10)
    out = th.nn.functional.sigmoid(th.randn([200, 10]))
    result = cross_entropy(label=label, out=out)
    true_ce = th.nn.functional.binary_cross_entropy(input=out, target=label.float())
    assert result.dim() == 0
    assert np.allclose(result, true_ce)


def test_net_output() -> None:
    """The network maps (BS, 1, 28, 28) to probabilities of shape (BS, 10)."""
    th.manual_seed(0)
    net = Net()
    out = net(th.randn(7, 1, 28, 28))
    assert out.shape == (7, 10)
    assert th.all((out > 0) & (out < 1))


def test_net_uses_conv_and_pooling() -> None:
    """The network has to be a CNN."""
    modules = list(Net().modules())
    assert any(isinstance(m, th.nn.Conv2d) for m in modules)
    assert any(isinstance(m, th.nn.MaxPool2d) for m in modules)


def _small_model() -> th.nn.Module:
    """A small model independent of Net, so these tests only check the optimizer."""
    th.manual_seed(0)
    return th.nn.Sequential(th.nn.Linear(5, 4), th.nn.ReLU(), th.nn.Linear(4, 3))


def test_sgd_step() -> None:
    """Every parameter is updated as p <- p - lr * grad."""
    model = _small_model()
    before = [p.detach().clone() for p in model.parameters()]
    for p in model.parameters():
        p.grad = th.randn_like(p)
    grads = [p.grad.detach().clone() for p in model.parameters()]
    model = sgd_step(model, learning_rate=0.1)
    for p, b, g in zip(model.parameters(), before, grads):
        assert th.allclose(p, b - 0.1 * g)


def test_zero_grad() -> None:
    """After zero_grad all gradients are zero."""
    model = _small_model()
    model(th.randn(8, 5)).sum().backward()
    assert any(th.any(p.grad != 0) for p in model.parameters())
    model = zero_grad(model)
    for p in model.parameters():
        assert p.grad is not None
        assert th.all(p.grad == 0)


class _FixedPredictor(th.nn.Module):
    """Predicts a fixed class and records whether gradients were enabled."""

    def __init__(self, cls: int) -> None:
        super().__init__()
        self.cls = cls
        self.grad_enabled = []

    def forward(self, x: th.Tensor) -> th.Tensor:
        self.grad_enabled.append(th.is_grad_enabled())
        preds = th.zeros(x.shape[0], 10)
        preds[:, self.cls] = 1.0
        return preds


def test_get_acc() -> None:
    """Accuracy of a constant predictor, computed without gradients."""
    imgs = th.zeros(40, 1, 28, 28)
    labels = th.tensor([3] * 10 + [5] * 30)
    loader = th.utils.data.DataLoader(
        th.utils.data.TensorDataset(imgs, labels), batch_size=20, shuffle=False
    )
    model = _FixedPredictor(cls=3)
    acc = get_acc(model=model, dataloader=loader)
    assert isinstance(acc, float)
    assert np.isclose(acc, 0.25)
    assert model.grad_enabled and not any(model.grad_enabled)


def test_training_reduces_loss() -> None:
    """A few SGD steps on a fixed batch reduce the loss."""
    th.manual_seed(0)
    net = Net()
    imgs = th.randn(32, 1, 28, 28)
    labels = th.nn.functional.one_hot(th.randint(0, 10, (32,)), num_classes=10)
    losses = []
    for _ in range(20):
        loss = cross_entropy(label=labels, out=net(imgs))
        loss.backward()
        net = sgd_step(net, learning_rate=0.5)
        net = zero_grad(net)
        losses.append(loss.item())
    assert losses[-1] < losses[0]
