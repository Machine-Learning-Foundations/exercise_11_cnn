"""Plot mnist digits."""

import matplotlib.pyplot as plt
import numpy as np

from mnist import get_mnist_train_data

# import tikzplotlib


if __name__ == "__main__":
    img_data_train, lbl_data_train = get_mnist_train_data()
    number_squence = img_data_train[0, :, :]

    for i in range(7):
        number_squence = np.concatenate(
            [number_squence, img_data_train[i + 1, :, :]], axis=1
        )

    plt.imshow(number_squence)
    plt.axis("off")
    # tikzplotlib.save("mnist_sequence.tex", standalone=True)
    plt.show()
