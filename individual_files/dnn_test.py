import os
import pathlib

from lib.dnn import Dnn
from lib.helper import Helper
from lib.model import Model
from lib.plotter import Plotter

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = Helper.load_data(
        os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "lib/mnist.npz")
    )
    input_dim = train_images.shape[1] * train_images.shape[2]
    class_num = train_labels.shape[1]
    test_samples, test_targets = Helper.get_random_samples(test_images, test_labels, 2000)
    train_samples, train_targets = Helper.get_random_samples(train_images, train_labels, 2000)

    test_samples = Model.flatten(test_samples)
    train_samples = Model.flatten(train_samples)

    dnn = Dnn(784, 10)
    loss, acc, confusion_matrices = dnn.train(
        train_samples, train_targets,
        test_samples, test_targets,
        batch_size=100, epoch=5
    )
    for cm in confusion_matrices:
        Plotter.plot_cm(cm).show()
    print(f"Accuracy={acc}")
    print(f"Loss={loss}")
