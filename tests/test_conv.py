"""Test the python function from src."""

import sys

import numpy as np
import pytest
import torch as th
from scipy.signal import correlate2d

sys.path.insert(0, "./src/")

from src.custom_conv import get_indices, my_conv, my_conv_direct

IMG_SHAPES = [(6, 6), (11, 10), (10, 11)]
KERNEL_SHAPES = [(2, 2), (3, 3), (2, 3), (3, 2), (4, 4)]


def _images(img_shape: tuple) -> list:
    """Return a structured (identity) and a random test image."""
    rng = np.random.default_rng(42)
    return [
        th.from_numpy(np.eye(*img_shape)),
        th.from_numpy(rng.normal(size=img_shape)),
    ]


def _kernel(kernel_shape: tuple) -> th.Tensor:
    rng = np.random.default_rng(0)
    return th.from_numpy(rng.uniform(0, 1, kernel_shape))


def _skip_if_get_indices_missing() -> None:
    """Skip the tests of the optional Task 1.2 until get_indices is implemented."""
    idx_list, _, _ = get_indices(th.zeros(3, 3), th.zeros(2, 2))
    if idx_list is None:
        pytest.skip("Optional Task 1.2: get_indices is not implemented yet.")


@pytest.mark.parametrize("img_shape", IMG_SHAPES)
@pytest.mark.parametrize("kernel_shape", KERNEL_SHAPES)
def test_conv(img_shape: tuple, kernel_shape: tuple) -> None:
    """Test the direct convolution code."""
    kernel = _kernel(kernel_shape)
    for img in _images(img_shape):
        my_res = my_conv_direct(img, kernel)
        res = correlate2d(img, kernel, mode="valid")
        assert my_res.shape == res.shape
        assert np.allclose(my_res, res)


def test_conv_kernel_equals_image() -> None:
    """A kernel of image size yields a single value, the full inner product."""
    img = th.arange(12, dtype=th.float64).reshape(3, 4)
    kernel = th.ones(3, 4, dtype=th.float64)
    my_res = my_conv_direct(img, kernel)
    assert my_res.shape == (1, 1)
    assert np.isclose(float(my_res[0, 0]), 66.0)


def test_get_indices_readme_example() -> None:
    """Check the index transformation from the README."""
    _skip_if_get_indices_missing()
    idx, rows, cols = get_indices(th.zeros(3, 3), th.zeros(2, 2))
    expected = np.array([[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7], [4, 5, 7, 8]])
    assert (rows, cols) == (2, 2)
    assert np.array_equal(np.asarray(idx), expected)


@pytest.mark.parametrize("img_shape", IMG_SHAPES)
@pytest.mark.parametrize("kernel_shape", KERNEL_SHAPES)
def test_conv_fast(img_shape: tuple, kernel_shape: tuple) -> None:
    """Test the convolution by matrix multiplication."""
    _skip_if_get_indices_missing()
    kernel = _kernel(kernel_shape)
    for img in _images(img_shape):
        my_res = my_conv(img, kernel)
        res = correlate2d(img, kernel, mode="valid")
        assert my_res.shape == res.shape
        assert np.allclose(my_res, res)
