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

    corr_rows = image_rows - kernel_rows + 1
    corr_cols = image_cols - kernel_cols + 1

    indices_x = torch.zeros((1, kernel_cols), dtype=torch.int)
    indices_x = indices_x + torch.unsqueeze(
        torch.arange(0, kernel_rows, dtype=torch.int), 1
    )
    indices_x = torch.stack([indices_x] * corr_cols)
    indices_x = torch.unsqueeze(indices_x, 0) + torch.arange(corr_rows).reshape(
        corr_rows, 1, 1, 1
    )
    indices_x = indices_x.reshape(corr_cols * corr_rows, kernel_rows, kernel_cols)

    indices_y = torch.zeros((kernel_rows, 1), dtype=torch.int)
    indices_y = indices_y + torch.unsqueeze(
        torch.arange(0, kernel_cols, dtype=torch.int), 0
    )
    indices_y = torch.unsqueeze(indices_y, 0) + torch.arange(corr_cols).reshape(
        corr_cols, 1, 1
    )
    indices_y = torch.concatenate([indices_y] * corr_rows)

    indices = torch.stack([indices_x, indices_y], 0).reshape(
        2, corr_cols * corr_rows, -1
    )
    idx_list = np.ravel_multi_index(indices.numpy(), (image_rows, image_cols), mode="clip")  # type: ignore
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
    for img_row in range(image_rows - kernel_rows + 1):
        corr_row = []
        for img_col in range(image_cols - kernel_cols + 1):
            img = image[
                img_row : (img_row + kernel_rows), img_col : (img_col + kernel_cols)
            ]
            res = torch.sum(img * kernel)
            corr_row.append(res)
        corr.append(torch.stack(corr_row))
    return torch.stack(corr)
