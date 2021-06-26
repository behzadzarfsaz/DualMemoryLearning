import logging
import io
import os
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from cv2 import cv2
from mpl_toolkits.axes_grid1 import ImageGrid
import traceback
from scipy import interpolate
from lib.som import SOM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Plotter")


class Plotter:
    def __init__(self, image_width, image_height):
        self.bmu_matrix = None
        self.image_width = image_width
        self.image_height = image_height

    def plot_som_main(self, som: SOM, images, labels):
        weights = som.get_weights()
        bmu_matrix = Plotter.generate_bmu_matrix(som, images)
        fig = plt.figure(figsize=(24.0, 8.0), facecolor='white')
        ax1 = fig.add_subplot(1, 3, 1, frameon=False, xticks=[], yticks=[])
        ax1.imshow(self.plot_som_weights(som, weights, image=True))
        ax2 = fig.add_subplot(1, 3, 2, frameon=False, xticks=[], yticks=[])
        ax2.imshow(self.plot_som_bmu(som, bmu_matrix, images))
        ax3 = fig.add_subplot(1, 3, 3, frameon=False, xticks=[], yticks=[])
        ax3.imshow(Plotter.plot_som_labels(som, bmu_matrix, labels))
        fig.tight_layout()
        return plt

    @staticmethod
    def generate_bmu_matrix(som, images):
        matrix = som.get_bmu_indexes(images)
        return np.reshape(
            matrix, newshape=[som.width, som.height]
        )

    def plot_som_weights(self, som, weights, image=True):
        fig = plt.figure(figsize=(8., 8.))
        grid = ImageGrid(
            fig, 111,
            nrows_ncols=(som.width, som.height),
            axes_pad=0.0,
        )
        weights_reshaped = np.reshape(
            weights, newshape=[-1, self.image_width, self.image_height]
        )
        for ax, im in zip(grid, list(weights_reshaped)):
            ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(im)
        if image:
            fig.suptitle('Weights Matrix', fontsize=20, fontweight='bold')
            return Plotter.get_img_from_fig(fig)
        return plt

    @staticmethod
    def plot_som_labels(som, bmu_matrix, labels):
        fig = plt.figure(figsize=(8., 8.))
        ax = fig.add_subplot(111)
        for i in range(som.width):
            for j in range(som.height):
                label = np.argmax(labels[bmu_matrix[i, j]])
                ax.text(
                    j + 0.5, som.height - i - 0.5, str(label),
                    fontdict={'weight': 'bold', 'size': 16}
                )
        ax.axis([0, som.width, 0, som.height])
        fig.suptitle('Mapped Labels', fontsize=20, fontweight='bold')
        return Plotter.get_img_from_fig(fig)

    def plot_som_changes(self, soms, som_samples, som_labels):
        w = soms[0].width
        h = soms[0].height
        bms = []
        l_diff = []
        diffs = []
        som_plots = []
        for i in range(len(soms)):
            bms.append(Plotter.generate_bmu_matrix(soms[i], som_samples[i]))
        for j in range(len(soms)):
            l_diff.append(np.argmax(som_labels[j][bms[j].ravel()], axis=1).reshape(w, h))
        for j in range(1, len(soms)):
            diffs.append(np.argwhere(l_diff[j] != l_diff[j - 1]).tolist())
        som_plots.append(self.plot_som_bmu(soms[0], bms[0], som_samples[0], main=False))
        for j in range(1, len(soms)):
            som_plots.append(
                self.plot_som_bmu(soms[j], bms[j], som_samples[j], diffs[j - 1], main=False)
            )
        for k in range(0, len(soms)):
            fig = plt.figure(figsize=(8.0, 8.0), facecolor='white')
            ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
            ax.imshow(som_plots[k])
            fig.tight_layout()
        return plt

    def plot_som_bmu(self, som, bmu_matrix, images, diffs=None, main=True):
        fig = plt.figure(figsize=(8.0, 8.0), facecolor='white')
        cnt = 0
        for i in (range(som.width)):
            for j in range(som.height):
                ax = fig.add_subplot(som.width, som.height, cnt + 1, frameon=False, xticks=[], yticks=[])
                if diffs is not None and [i, j] in diffs:
                    cc = 'hot'
                else:
                    cc = 'Greys'
                ax.imshow(
                    images[bmu_matrix[i, j]].reshape([self.image_width, self.image_height]),
                    cmap=cc, interpolation='nearest'
                )
                cnt = cnt + 1
        if main:
            fig.suptitle("BMU's Matrix", fontsize=20, fontweight="bold")
        return Plotter.get_img_from_fig(fig)

    @staticmethod
    def get_img_from_fig(fig, dpi=300):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.close()
        return img

    @staticmethod
    def plot_evaluation(accuracy_values, epoch_size):
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figsize=(18, 10))
        ax = fig.add_subplot(111, projection='3d')
        xs = np.arange(0, epoch_size)
        verts = []
        colors = ['b', 'g', 'c', 'y', 'r']
        zs = [int(i) for i in range(len(accuracy_values))]
        for acc, z in enumerate(zs):
            ys = accuracy_values[acc]
            ys[0], ys[-1] = 0, 0
            verts.append(list(zip(xs, ys)))
        for v in range(len(verts)):
            props = dict(boxstyle='round', facecolor=colors[v], alpha=0.5)
            t = f" D{v + 1} Avg.:{np.mean(accuracy_values[v]):.2f}"
            ax.text2D(x=0.0, y=v * 0.05, s=t, transform=ax.transAxes, bbox=props)
        poly = PolyCollection(verts, facecolors=[colors[c] for c in range(len(accuracy_values))])
        poly.set_alpha(0.8)
        ax.add_collection3d(poly, zs=zs, zdir='y')
        ax.set_xlabel('Test Samples', labelpad=30)
        ax.set_xlim3d(0, epoch_size)
        ax.set_ylabel('Evaluation', labelpad=30)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(["D1", "D2", "D3", "D4", "D5"])
        ax.set_zlabel('Accuracy', labelpad=30)
        ax.set_zlim3d(0, 1.0)
        fig.tight_layout()
        return plt

    @staticmethod
    def plot_dnn(accuracy_values, epoch_size, subtask):
        a = np.array(accuracy_values).ravel()
        task_num = a.shape[0] // epoch_size
        if task_num > 2:
            plt.rcParams.update({'font.size': 40})
            fig, ax = plt.subplots(figsize=(task_num * 10, 8))
        else:
            plt.rcParams.update({'font.size': 20})
            fig, ax = plt.subplots()
        a[0:10] = a[11]
        max_acc = a[-1]

        cnt = 1
        for i in range(0, a.shape[0], epoch_size):
            ax.axvline(x=i, lw=5, color='black', zorder=5)
            if task_num > 1:
                ax.text(i + epoch_size / 3, 0.1, f"Loop {cnt}", fontsize=25)
            if cnt % 2 == 0:
                ax.axvspan(i, i + epoch_size, facecolor='seagreen', alpha=0.3)
            if cnt == a.shape[0] // epoch_size:
                ax.text(i + epoch_size / 3, 0.4, subtask, fontsize=80, color='seagreen', alpha=0.7)
            cnt += 1
        ax.set_xlim(0, a.shape[0])
        labels = ['0']
        intervals = list(range(0, a.shape[0], epoch_size))
        for _ in range(len(intervals) - 1):
            labels.append(f"{epoch_size}/0")
        intervals.append(a.shape[0])
        labels.append(str(epoch_size))
        plt.xticks(intervals, labels)
        ax.set_ylim(0, 1)
        ax.plot(a, linewidth=5)
        plt.grid(True)
        plt.ylim(top=1, bottom=0)
        plt.ylabel(f'Last={max_acc:.2f}')
        fig.tight_layout()
        return plt

    @staticmethod
    def plot_cm(cm):
        fig, ax = plt.subplots()
        ax.grid(True)
        fig.gca().invert_yaxis()
        for row in range(0, cm[1].shape[0]):
            ax.text(row + 0.4, -0.3, str(row))
            ax.text(-0.3, row + 0.6, str(row))
            for col in range(1, cm[1].shape[1] + 1):
                if row == col - 1:
                    face_color = 'red'
                else:
                    face_color = 'black'
                ax.text(
                    row + 0.1, col - 0.33, str(cm[1][col - 1, row]),
                    fontdict={'size': 12}, color=face_color
                )

        ax.set_xlabel('Targets', labelpad=5)
        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ax.set_ylabel('Predicts', labelpad=10)
        fig.suptitle(cm[0])
        return plt

    @staticmethod
    def plot_cm_diagram(cm_list, classes=None):
        plt.rcParams.update({'font.size': 15})
        if classes is None:
            classes = range(10)
        fig, ax = plt.subplots()
        ax.grid(True)
        corrects = []
        for cm in cm_list:
            corrects.append(cm.diagonal())
        corrects = np.array(corrects).astype("int32")
        x = list(range(0, corrects.shape[0], 10))
        x.append(corrects.shape[0] - 1)
        x = np.array(x).astype("int32")
        for s in classes:
            y = corrects[:, s][x]
            t, c, k = interpolate.splrep(x, y, s=0, k=4)
            x_smooth = np.linspace(x.min(), x.max(), len(cm_list))
            spline = interpolate.BSpline(t, c, k, extrapolate=False)
            syy = spline(x_smooth)
            syy[syy < 5] = 0.0
            plt.plot(x_smooth, syy, linewidth=2, label=f"Class {s}")
        ax.legend()
        return plt

    @staticmethod
    def plot_sigma(sigma_list):
        fig, ax = plt.subplots()
        legends = []
        ax.grid(True)
        fig.suptitle('SOM Sigma Decay')
        for idx, s in enumerate(sigma_list):
            ax.plot(np.array(s), linewidth=3)
            legends.append(f"Sigma D{idx + 1}")
        ax.legend(legends)
        return plt

    @staticmethod
    def save_plot(plot, img_path, file_name, extension="pdf"):
        try:
            path = os.path.join(img_path, file_name) + "." + extension
            plot.savefig(path)
            logger.info(f'File {path} saved.')
        except IOError as ex:
            logger.error(ex)
            logger.error(traceback.format_exc())
