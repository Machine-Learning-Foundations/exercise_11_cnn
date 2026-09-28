"""Plot mnist digits."""

import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as tvd

# import tikzplotlib


if __name__ == "__main__":
    dataset = tvd.MNIST("./.data", train=True, download=True)
    img_data_train = dataset.data.numpy()  # (60000, 28, 28), uint8
    lbl_data_train = dataset.targets.numpy()  # (60000,)

    number_sequence = np.concatenate(list(img_data_train[:8]), axis=1)
    print("labels:", lbl_data_train[:8])

    plt.imshow(number_sequence)
    plt.axis("off")
    # tikzplotlib.save("mnist_sequence.tex", standalone=True)
    plt.show()
