"""Test the python functions from src/mnist."""

import sys

import numpy as np
import torch as th

sys.path.insert(0, "./src/")

from src.mnist import cross_entropy


def test_cross_entropy() -> None:
    """Test if the cross entropy is implemented correctly."""
    label = th.nn.functional.sigmoid(th.randn([200, 10]))
    out = th.nn.functional.sigmoid(th.randn([200, 10]))
    result = cross_entropy(label=label, out=out)
    true_ce = th.nn.functional.binary_cross_entropy(input=out, target=label)
    assert np.allclose(result, true_ce)
