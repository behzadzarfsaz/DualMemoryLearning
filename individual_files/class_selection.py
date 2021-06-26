import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from lib.helper import Helper

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = Helper.load_data(
        os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "lib/mnist.npz")
    )

    class_5_idx = np.argwhere(np.argmax(train_labels, axis=1) == 5).ravel()
    r = class_5_idx[np.random.randint(0, class_5_idx.shape[0], 5)]
    sample_5 = train_images[r]
    plt.figure(figsize=(15, 5))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(sample_5[i].reshape([28, 28]))
        plt.axis('off')
    plt.subplots_adjust(wspace=0.3, hspace=-0.1)
    plt.show()
