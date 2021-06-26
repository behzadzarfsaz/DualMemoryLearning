import copy
import json
import logging
from optparse import OptionParser
from lib.model import Model
import numpy as np
import os
import re
import pathlib
from lib.plotter import Plotter
from lib.helper import Helper
import matplotlib.pyplot as plt
import traceback

parser = OptionParser()
parser.add_option("--batch", dest="batch_size",
                  help="The size of batches for training.", default=1, type=int)
parser.add_option("--x", dest="x",
                  help="SOM x dimension.", default=10, type=int)
parser.add_option("--y", dest="y",
                  help="SOM y dimension.", default=10, type=int)
parser.add_option("--radius", dest="radius",
                  help="Initial radius of SOM training.",
                  default='2.0,0.07,0.07,0.07,0.07', type=str)
parser.add_option("--lr", dest="lr",
                  help="Initial learning rate of SOM training.",
                  default='0.5,0.02,0.02,0.02,0.02 ', type=str)
parser.add_option("--dnn_iter", dest="dnn_iter",
                  help="DNN iteration for each batch.",
                  default='1,1,1,1,1', type=str)
parser.add_option("--epoch", dest="epoch",
                  help="Epoch for each sub-task.",
                  default='1,1,1,1,1', type=str)
parser.add_option("--d1", dest="d1",
                  help="The list of samples to train in part 1.", default=None, type=str)
parser.add_option("--d2", dest="d2",
                  help="The list of samples to train in part 2.", default=None, type=str)
parser.add_option("--d3", dest="d3",
                  help="The list of samples to train in part 3.", default=None, type=str)
parser.add_option("--d4", dest="d4",
                  help="The list of samples to train in part 4.", default=None, type=str)
parser.add_option("--d5", dest="d5",
                  help="The list of samples to train in part 2.", default=None, type=str)
parser.add_option("--limit", dest="limit",
                  help="Limiting the number of samples for each class. ", default=1000, type=int)
parser.add_option("--ce", dest="ce",
                  help="Threshold cross-entropy. ", default=0.2, type=float)
parser.add_option("--stm", dest="stm",
                  help="Short-term memory size. ", default=100, type=int)
parser.add_option("--image_path", dest="img_path",
                  help="Path for saving SOM images.", default=None, type=str)
parser.add_option("--plot_som", action="store_true", default=False, help="Plot the generated SOMs.")

(options, args) = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Main")

if __name__ == "__main__":

    # ===================== Instantiating =======================
    options_dict = options.__dict__
    train_images, train_labels, test_images, test_labels = Helper.load_data(
        os.path.join(pathlib.Path(__file__).parent.absolute(), "lib/mnist.npz")
    )
    input_dim = train_images.shape[1] * train_images.shape[2]
    class_num = train_labels.shape[1]
    di = [int(l) for l in options.dnn_iter.split(',')]
    epochs = [int(l) for l in options.epoch.split(',')]
    rad = [float(l) for l in options.radius.split(',')]
    lr = [float(l) for l in options.lr.split(',')]
    x_test, t_test = Helper.get_random_samples(
        test_images, test_labels, options.limit
    )
    model = Model(
        input_dim, options.batch_size, options.x,
        options.y, class_num, Model.flatten(x_test), t_test,
        options.limit, options.stm
    )

    try:
        # ===================== Generating sub-sets =======================
        sub_sets = []
        for opt in sorted(options_dict.keys()):
            if re.search(r"^d[0-5]$", opt):
                if options_dict[opt] is not None:
                    classes = [int(n) for n in options_dict[opt].split(',')]
                    sub_sets.append(
                        Helper.get_class_samples(train_images, train_labels, classes, options.limit)
                    )
        # ===================== Training =======================
        samples_so_far = None
        labels_so_far = None
        accuracy_values = []
        soms = []
        som_samples = []
        som_labels = []
        sigma = []
        confusion_matrices = []
        fl_matrices = []
        for num, sub_set in enumerate(sub_sets):
            if num > 0:
                samples_so_far = np.concatenate([samples_so_far, sub_set['samples']])
                labels_so_far = np.concatenate([labels_so_far, sub_set['labels']])
            else:
                samples_so_far = sub_set['samples']
                labels_so_far = sub_set['labels']

            # ======================== Train =========================
            accuracy, c_sigma, c_matrices = model.train(
                Model.flatten(sub_set['samples']), sub_set['labels'], di[num], lr[num],
                rad[num], options.ce, num + 1, epochs[num]
            )

            # ================ Collecting Plot Data =================
            accuracy_values.append(accuracy)
            sigma.append(c_sigma)
            for c in c_matrices:
                confusion_matrices.append(c)

            fl_matrices.append((f"D{num + 1}|CM{1}", c_matrices[0]))
            fl_matrices.append((f"D{num + 1}|CM{2}", c_matrices[-1]))

            soms.append(copy.copy(model.som))
            som_rs, som_rt = Helper.get_random_samples(samples_so_far, labels_so_far, 1000)
            som_samples.append(model.flatten(som_rs))
            som_labels.append(som_rt)

        # ================== Plotting Sigma Decay ====================
        Plotter.plot_sigma(sigma)
        # ============== Plotting Confusion Matrices =================
        for cm in fl_matrices:
            Plotter.plot_cm(cm)
        Plotter.plot_cm_diagram(confusion_matrices)
        # ===================== Plotting SOM's =======================
        if options.plot_som:
            # Plotter(28, 28).plot_som_changes(soms, som_samples, som_labels)
            for s in soms:
                Plotter(28, 28).plot_som_weights(s, s.get_weights(), image=False)

        # ===================== Plotting 3D Evaluation =======================
        flatten_acc = []
        for a in accuracy_values:
            temp = np.array([i for i in a])
            flatten_acc.append(temp.flatten())
        Plotter.plot_evaluation(flatten_acc, x_test.shape[0])

        # ===================== Save/Show Plots =======================
        if options.img_path:
            for i in plt.get_fignums():
                pl = plt.figure(i)
                Plotter.save_plot(pl, options.img_path, f"plot{i}", "pdf")
                with open(os.path.join(options.img_path, 'params.txt'), 'w') as f:
                    f.write(json.dumps(options_dict, indent=2))
        else:
            for i in plt.get_fignums():
                pl = plt.figure(i)
                pl.show()

    except Exception as ex:
        logger.error(ex)
        logger.error(traceback.format_exc())
