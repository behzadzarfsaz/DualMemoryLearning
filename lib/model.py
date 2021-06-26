import copy
import logging
import shutil
from math import exp
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.utils import shuffle
from tqdm import trange
from lib.bqueue import Bqueue
from lib.dnn import Dnn
from lib.helper import Helper
from lib.som import SOM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Model")
terminal_columns = shutil.get_terminal_size().columns // 2


class Model:
    def __init__(
            self, input_dim, batch_size, som_x, som_y,
            label_class_num, xt, tt, limit, stm
    ):
        self.input_dim = input_dim
        self.som_x = som_x
        self.som_y = som_y
        self.class_num = label_class_num
        self.batch_size = batch_size
        self.som = SOM(self.som_x, self.som_y, self.input_dim)
        self.dnn = Dnn(self.som_x * self.som_y, self.class_num)
        self.x_test = xt
        self.t_test = tt
        self.stm = Bqueue(max_size=stm)
        self.limit = limit
        self.scaler = StandardScaler()
        self.max = 1.0

    def transfer(self, dist, test=False):
        if self.max < np.max(dist) and not test:
            self.max = np.max(dist)
        dist /= self.max
        return self.scaler.fit_transform(dist)

    @staticmethod
    def flatten(samples):
        return np.reshape(samples, newshape=[-1, samples.shape[1] * samples.shape[2]])

    def reply(self):
        samples = None
        labels = None
        stm_samples = np.array([s[0] for s in self.stm.get_list()]).astype("float32")
        stm_labels = np.array([s[1] for s in self.stm.get_list()]).astype("float32")
        if stm_samples.shape[0] > 0:
            for i in trange(self.class_num, desc="Replaying Data"):
                class_stm_idx = np.argwhere(np.argmax(stm_labels, axis=1) == i).ravel()
                if class_stm_idx.shape[0] == 0:
                    break
                class_prototypes = stm_samples[class_stm_idx]
                ll = stm_labels[class_stm_idx]
                g_samples = np.repeat(
                    class_prototypes, self.limit // class_prototypes.shape[0], axis=0
                )
                g_labels = np.repeat(ll, self.limit // class_prototypes.shape[0], axis=0)
                if i == 0:
                    samples = g_samples
                    labels = g_labels
                else:
                    samples = np.concatenate((samples, g_samples))
                    labels = np.concatenate((labels, g_labels))
        return samples, labels

    def fill_stm(self, samples, z_som, labels):
        logger.info("\rFilling STM")
        _, acc = self.dnn.evaluate(z_som, labels, batch_size=1, verbose=0)
        acc = np.array(acc).astype("float32")
        stm_idx = np.argwhere(acc > 0.5).ravel()
        for s in range(self.class_num):
            class_idx = np.argwhere(np.argmax(labels[stm_idx], axis=1) == s).ravel()
            np.random.shuffle(class_idx)
            class_samples = samples[class_idx]
            class_labels = labels[class_idx]
            class_samples, class_labels = shuffle(class_samples, class_labels)
            loop_iter = min(self.stm.max_size // self.class_num, class_idx.shape[0])
            for i in range(loop_iter):
                self.stm.push(
                    (class_samples[i], class_labels[i])
                )

    def train(
            self, samples, labels, dnn_iter, som_lr, som_rad, ce, sub_task, epoch
    ):
        samples, labels = shuffle(samples, labels)
        logger.info("\r".center(terminal_columns, "="))
        logger.info(f"\r Sub-Task D{sub_task}")
        logger.info("\r".center(terminal_columns, "="))
        confusion_matrices = []
        sigma = []
        r_samples = None
        r_labels = None
        if sub_task > 1 and self.stm.max_size > 0:
            m_samples, m_labels = self.reply()
            if m_samples is not None:
                r_samples = np.concatenate((samples, m_samples))
                r_labels = np.concatenate((labels, m_labels))
                r_samples, r_labels = shuffle(r_samples, r_labels)
        else:
            r_samples = samples
            r_labels = labels

        for ep, e in enumerate(range(epoch)):
            new_labels = np.unique(np.argmax(labels, axis=1))
            x, t = Helper.generate_batches(r_samples, r_labels, self.batch_size)
            sigma = []
            d_acc = 0.0
            cm_list = range(len(x))
            pbar = trange(len(x))
            d_counter = 0
            for i in pbar:
                z_som = self.transfer(self.som.get_distances(x[i]), test=True)
                loss, acc = self.dnn.evaluate(z_som, t[i], verbose=0)
                loss = np.array(loss)
                wrong_idx = np.argwhere(np.greater(np.array(loss), ce)).ravel()
                if wrong_idx.shape[0] > 0:
                    decay = exp(-1 * ((10 / sub_task) * d_counter / len(x)))
                    sigma.append(som_rad * decay)
                    d_counter += 1
                    mask = np.isin(np.argmax(t[i][wrong_idx], axis=1), new_labels)
                    new_wrong_samples = x[i][wrong_idx][mask]
                    self.som.train(
                        new_wrong_samples, learning_rate=som_lr * decay,
                        radius=som_rad * decay, global_order=self.batch_size
                    )
                    z_som = self.transfer(
                        self.som.get_distances(x[i], batch_size=self.batch_size)
                    )
                    z_som_test = self.transfer(
                        self.som.get_distances(self.x_test, batch_size=self.batch_size), test=True
                    )
                    cm = i in cm_list
                    d_loss, d_acc, confusion_matrix = self.dnn.train(
                        z_som, t[i], z_som_test, self.t_test,
                        cm=cm, epoch=dnn_iter, batch_size=self.batch_size
                    )
                    if len(confusion_matrix) > 0:
                        for m in confusion_matrix:
                            confusion_matrices.append(m)
                    d_acc = np.mean(np.array(d_acc).astype("float32"))
                else:
                    confusion_matrices.append(copy.copy(confusion_matrices[-1]))
                pbar.set_description(
                    f"Epoch{ep + 1}/{epoch}"
                    f"|Batch:{i + 1}/{len(x)}"
                    f"|CE:{wrong_idx.shape[0]}/{x[i].shape[0]}"
                    f"|Train Acc.:{d_acc:.4f}"
                )
                pbar.refresh()
        logger.info("\rEvaluation...")
        z_som_test = self.transfer(self.som.get_distances(self.x_test, batch_size=self.batch_size), test=True)
        z_som_stm = self.transfer(self.som.get_distances(r_samples, batch_size=self.batch_size), test=True)
        loss, accuracy = self.dnn.evaluate(z_som_test, self.t_test, verbose=1)
        if self.stm.max_size > 0:
            self.fill_stm(r_samples, z_som_stm, r_labels)
        return accuracy, np.array(sigma), confusion_matrices
