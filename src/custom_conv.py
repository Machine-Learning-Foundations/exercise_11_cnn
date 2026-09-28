"""This module ships a function."""

import numpy as np
import torch


def get_indices(image: torch.Tensor, kernel: torch.Tensor) -> tuple:
    """Get the indices to set up pixel vectors for convolution by matrix-multiplication.

    Args:
        image (torch.Tensor): The input image of shape [height, width].
        kernel (torch.Tensor): A 2d-convolution kernel.

    Returns:
        tuple: An integer array with the indices, the number of rows in the result,
        and the number of columns in the result.
    """
    image_rows, image_cols = image.shape
    kernel_rows, kernel_cols = kernel.shape

    # (Optional) 1.2 TODO: Implement me
    # Hint: np.ravel_multi_index turns (row, col) index pairs into flat indices.
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

    This function implements a slow summation in a for loop.

    Args:
        image (torch.Tensor): The input image of shape [height, width].
        kernel (torch.Tensor): A 2d-convolution kernel.

    Returns:
        torch.Tensor: The cross-correlation of shape
        [height - kernel_rows + 1, width - kernel_cols + 1].
    """
    image_rows, image_cols = image.shape
    kernel_rows, kernel_cols = kernel.shape
    corr = []
    # 1.1.1 TODO: Implement direct convolution.
    return torch.tensor([0.0])
