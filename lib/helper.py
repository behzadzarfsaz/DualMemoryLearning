import glob
import os
import numpy as np


class Helper:
    @staticmethod
    def load_data(path):
        with np.load(path) as f:
            x_train, y_train = f['arr_0'], f['arr_2']
            x_test, y_test = f['arr_1'], f['arr_3']
            return x_train, y_train, x_test, y_test

    @staticmethod
    def delete_folder_content(path):
        file_list = glob.glob(os.path.join(path, "*.h5"))
        for f in file_list:
            os.remove(f)

    @staticmethod
    def get_random_samples(data, labels, num):
        idx = np.random.randint(0, data.shape[0], [num])
        return data[idx], labels[idx]

    @staticmethod
    def get_class_samples(data, labels, d_set, limit):
        all_idx = []
        for dset in d_set:
            idx = np.argwhere(np.argmax(labels, axis=1) == dset).ravel()
            if limit is not None:
                idx = idx[0:limit]
            all_idx.append(idx)
        all_idx = np.array(all_idx).ravel()
        return {
            "samples": data[all_idx],
            "labels": labels[all_idx]
        }

    @staticmethod
    def generate_batches(samples, labels, batch_size):
        idx = np.arange(0, samples.shape[0])
        x_batches = [samples[i: i + batch_size] for i in range(0, idx.shape[0], batch_size)]
        t_batches = [labels[i: i + batch_size] for i in range(0, idx.shape[0], batch_size)]
        return x_batches, t_batches
