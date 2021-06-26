import copy
import logging
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DNN")


class Dnn:
    def __init__(self, input_size, class_num):
        self.class_num = class_num
        self.input_size = input_size
        self.model = Sequential()
        self.model.add(BatchNormalization())
        self.model.add(
            Dense(
                input_size * 2, input_dim=input_size,
                activation='relu',
                kernel_initializer='uniform'
            )
        )
        self.model.add(BatchNormalization())
        self.model.add(
            Dense(
                class_num, input_dim=input_size * 2,
                activation='softmax',
                kernel_initializer='uniform'
            )
        )
        self.model.compile(
            optimizer=SGD(lr=0.005),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    def train(
            self, samples, labels, test_samples, test_labels,
            epoch=1, batch_size=1, cm=False
    ):
        matrices = []
        history = TrainHistory(self.model, epoch)
        self.model.fit(
            samples, labels, batch_size=batch_size, epochs=epoch,
            verbose=0, callbacks=[history]
        )
        if cm:
            cm_model = copy.copy(self.model)
            cm_labels = np.argmax(test_labels, axis=1)
            for weight in history.weights_list:
                confusion_matrix = np.zeros([self.class_num, self.class_num]).astype("int32")
                cm_model.set_weights(weight)
                predicts = np.argmax(
                    cm_model.predict(test_samples), axis=1
                ).astype("int32")
                for p in range(predicts.shape[0]):
                    confusion_matrix[predicts[p], cm_labels[p]] += 1
                matrices.append(confusion_matrix)
        return history.loss, history.acc, matrices

    def evaluate(self, xt, yt, batch_size=1, verbose=0):
        history = EvaluationHistory()
        self.model.evaluate(xt, yt, verbose=0, batch_size=batch_size, callbacks=[history])
        if verbose == 1:
            logger.info(f"\rTest Accuracy={np.mean(np.array(history.acc))}")
            logger.info(f"\rTest Loss={np.mean(np.array(history.loss))}")
        return history.loss, history.acc


class TrainHistory(Callback):
    def __init__(self, model, epochs):
        super().__init__()
        self.model_list = model
        self.weights_list = []
        self.loss = 0.0
        self.acc = 0.0
        self.epochs = epochs

    def on_train_begin(self, logs=None):
        self.weights_list = []
        self.loss = 0.0
        self.acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 == self.epochs:
            self.weights_list.append(self.model.get_weights())
        self.loss = logs.get('loss')
        self.acc = logs.get('accuracy')


class EvaluationHistory(Callback):
    def __init__(self):
        super().__init__()
        self.loss = []
        self.acc = []

    def on_test_begin(self, logs=None):
        self.loss = []
        self.acc = []

    def on_test_batch_end(self, batch, logs=None):
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
