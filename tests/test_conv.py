"""Test the python function from src."""

import sys

import numpy as np
import pytest
import torch as th
from scipy.signal import correlate2d

sys.path.insert(0, "./src/")

from src.custom_conv import my_conv, my_conv_direct


@pytest.mark.parametrize("img_shape", [(6, 6), (11, 10), (11, 10)])
@pytest.mark.parametrize("kernel_shape", [(2, 2), (3, 3), (2, 3), (4, 4)])
def test_conv(img_shape: tuple, kernel_shape: tuple) -> None:
    """Test the convolution code."""
    img = th.from_numpy(np.eye(*img_shape))
    kernel = th.from_numpy(np.array(np.random.uniform(0, 1, kernel_shape)))
    my_res = my_conv_direct(img, kernel)
    res = correlate2d(img, kernel, mode="valid")

    assert np.allclose(my_res, res)


@pytest.mark.parametrize("img_shape", [(6, 6), (11, 10), (11, 10)])
@pytest.mark.parametrize("kernel_shape", [(2, 2), (3, 3), (2, 3), (4, 4)])
def test_conv_fast(img_shape: tuple, kernel_shape: tuple) -> None:
    """Test the convolution code."""
    img = th.from_numpy(np.eye(*img_shape))
    kernel = th.from_numpy(np.array(np.random.uniform(0, 1, kernel_shape)))
    my_res = my_conv(img, kernel)
    res = correlate2d(img, kernel, mode="valid")

    assert np.allclose(my_res, res)
