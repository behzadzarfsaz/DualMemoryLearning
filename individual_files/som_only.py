import os
import pathlib
from lib.model import Model
from lib.plotter import Plotter
from lib.som import SOM
from lib.helper import Helper

if __name__ == "__main__":
    """
    The main entry point.
    """
    train_images, train_labels, test_images, test_labels = Helper.load_data(
        os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "lib/mnist.npz")
    )
    input_dim = train_images.shape[1] * train_images.shape[2]
    class_num = train_labels.shape[1]
    som = SOM(15, 15, input_dim)

    samples, labels = Helper.get_random_samples(train_images, train_labels, 10000)
    som.train(Model.flatten(samples), 0.5, radius=5.0, batch_size=10)
    plotter = Plotter(28, 28)
    plot_samples, plot_labels = Helper.get_random_samples(train_images, train_labels, 1000)
    img = plotter.plot_som_main(som, Model.flatten(plot_samples), plot_labels)
    img.show()
