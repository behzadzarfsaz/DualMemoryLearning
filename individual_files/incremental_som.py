import os
import pathlib
import matplotlib.pyplot as plt
from lib.helper import Helper
from lib.model import Model
from lib.plotter import Plotter
from lib.som import SOM

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = Helper.load_data(
        os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "lib/mnist.npz")
    )
    options_dict = {
        'd1': [0, 1],
        'd2': [2, 3],
        'd3': [4, 5],
        'd4': [6, 7],
        'd5': [8, 9]
    }
    lr = [0.01, 0.01, 0.01, 0.01, 0.01]
    r = [4.5, 4.5, 4.5, 4.5, 4.5]
    sub_sets = []
    for opt in sorted(options_dict.keys()):
        if options_dict[opt] is not None:
            sub_sets.append(
                Helper.get_class_samples(
                    train_images, train_labels, options_dict[opt], 1000
                )
            )

    som = SOM(10, 10, 784)
    plotter = Plotter(28, 28)
    for i, s in enumerate(sub_sets):
        for j in range(3):
            som.train(Model.flatten(s['samples']), learning_rate=lr[i], radius=r[i])
        plotter.plot_som_weights(som, som.get_weights(), image=False)

    for i in plt.get_fignums():
        pl = plt.figure(i)
        pl.show()
