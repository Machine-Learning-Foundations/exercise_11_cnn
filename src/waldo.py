"""Get the computer to find waldo."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from custom_conv import my_conv_direct

# from custom_conv import my_conv
# from scipy.signal import correlate2d


if __name__ == "__main__":
    problem_image = np.array(Image.open("./data/waldo/waldo_space.jpg"))
    waldo = np.array(Image.open("./data/waldo/waldo_small.jpg"))

    problem_image = np.mean(problem_image, -1)[1000:1500, 1000:1500]
    waldo = np.mean(waldo, -1)

    plt.imshow(waldo)
    plt.show()

    plt.imshow(problem_image)
    plt.show()

    # Normalizing images such that they have mean 0 and variance 1.
    mean = np.mean(problem_image)
    std = np.std(problem_image)
    problem_image = (problem_image - mean) / std
    waldo = (waldo - mean) / std

    # Our convolution functions expect torch tensors.
    problem_image_th = torch.from_numpy(problem_image)
    waldo_th = torch.from_numpy(waldo)

    # Selfmade direct convolution: slow (python loops).
    # 1.1.2 TODO: Use your own convolution function to find waldo.
    conv_res = my_conv_direct(problem_image_th, waldo_th).numpy()

    # Selfmade fast version (Optional Task 1.2): the index matrix has
    # (#output pixels) x (#kernel pixels) entries, which costs several GB here.
    # conv_res = my_conv(problem_image_th, waldo_th).numpy()

    # Built in function, very fast.
    # 1.1.3 TODO: Use scipy's correlate2d function to find waldo.
    # conv_res = correlate2d(problem_image, waldo, mode="valid", boundary="fill")

    max_idx = np.argmax(conv_res)
    idx = np.unravel_index(max_idx, conv_res.shape)
    print(idx)
    plt.imshow(np.log(np.abs(conv_res)))
    plt.plot(idx[1], idx[0], "x")
    plt.colorbar()
    plt.show()

    plt.imshow(problem_image)
    plt.show()
