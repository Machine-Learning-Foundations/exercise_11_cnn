"""This module ships a function."""

import numpy as np
import torch


def get_indices(image: torch.Tensor, kernel: torch.Tensor) -> tuple:
    """Get the indices to set up pixel vectors for convolution by matrix-multiplication.

    Args:
        image (jnp.ndarray): The input image of shape [height, width.]
        kernel (jnp.ndarray): A 2d-convolution kernel.

    Returns:
        tuple: An integer array with the indices, the number of rows in the result,
        and the number of columns in the result.
    """
    image_rows, image_cols = image.shape
    kernel_rows, kernel_cols = kernel.shape

    # TODO: Implement me
    idx_list = None
    corr_rows = None
    corr_cols = None 
    return idx_list, corr_rows, corr_cols


def my_conv(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Evaluate a selfmade convolution function.

    This function implements the summation via matrix multiplication.
    """
    idx_list, corr_rows, corr_cols = get_indices(image, kernel)
    img_vecs = image.flatten()[idx_list]
    corr_flat = img_vecs @ kernel.flatten()
    corr = corr_flat.reshape(corr_rows, corr_cols)
    return corr


def my_conv_direct(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Evaluate a selfmade convolution function.

    Thus function implements very slow summation in a for loop.
    """
    image_rows, image_cols = image.shape
    kernel_rows, kernel_cols = kernel.shape
    corr = []
    # TODO: Implement direct convolution.
    return torch.tensor([0.])
