import itertools
import logging
import tensorflow as tf
import numpy as np
from tqdm import trange, tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SOM")


class SOM:
    def __init__(self, x, y, input_dim):
        self.width = x
        self.height = y
        self.input_dim = input_dim
        self.qe = 0.
        self.weights = tf.Variable(
            tf.random.uniform([self.width * self.height, self.input_dim])
        )
        self.lox = None
        self.loy = None

    @staticmethod
    @tf.function
    def do_step(
            input_vector, each_batch_size, weights, loc_x, loc_y, x, y,
            input_dim, current_batch_num, batch_num, go, learning_rate,
            radius
    ):
        # Broadcasting weights to batch size.
        weights_batch = SOM.broadcast_to_batch(
            weights, each_batch_size, input_dim, x, y, 0)
        # Broadcasting input vector to (som.width x som.height).
        input_reshaped = SOM.broadcast_to_batch(
            input_vector, each_batch_size, input_dim, x, y, 1)

        # Calculating The decay function for reducing the learning rate and radius by current batch number.
        # Also can maintain the decay fixed in case of relieving radius and lr from caller function.
        decay = tf.cond(
            tf.less(current_batch_num, go),
            lambda: 1.0,
            lambda: tf.subtract(
                1.0, tf.cast(
                    tf.divide(
                        tf.subtract(current_batch_num, go),
                        batch_num
                    ), dtype=tf.float32
                )
            )
        )

        # Calculating learning rate.
        current_learning_rate = tf.multiply(learning_rate, decay)

        # Calculating radius.
        current_radius = tf.multiply(radius, decay)

        # Calculating Euclidean BMu distance of input vector.
        bmu_distance = SOM.get_bmu_distance(
            loc_x, loc_y, each_batch_size, x, y, input_reshaped, weights_batch
        )

        # Calculating neighborhood function.
        neighborhood = tf.reshape(
            SOM.get_neighborhood(
                bmu_distance, current_radius, current_learning_rate),
            shape=[each_batch_size, x * y]
        )

        # Broadcasting neighborhood function to input dimension.
        neighborhood_compatible = tf.broadcast_to(
            tf.expand_dims(neighborhood, 2),
            shape=[each_batch_size, x * y, input_dim]
        )

        # Calculating the difference of new weights.
        delta = SOM.calculate_delta(
            neighborhood_compatible, input_reshaped, weights_batch
        )

        # Applying the difference (Delta) to current weights.
        new_weights = tf.reduce_mean(tf.add(weights_batch, delta), axis=0)
        return new_weights

    @staticmethod
    def broadcast_to_batch(input_tensor, batch_size, input_dim, x, y, axis=0):
        return tf.broadcast_to(
            tf.expand_dims(input_tensor, axis=axis),
            [batch_size, x * y, input_dim]
        )

    @staticmethod
    def calculate_bmu(input_reshaped, weights_batch):
        distances = tf.reduce_sum(
            tf.pow(
                tf.subtract(input_reshaped, weights_batch), 2
            ),
            axis=2
        )
        return tf.argmin(distances, axis=1)

    @staticmethod
    def get_neighborhood(bmu_distance, current_radius, current_learning_rate):
        neighbourhood_function = tf.exp(
            tf.negative(
                tf.divide(
                    tf.cast(
                        bmu_distance,
                        dtype=tf.float32
                    ),
                    tf.cast(
                        tf.pow(current_radius, 2),
                        dtype=tf.float32
                    )
                )
            )
        )
        return tf.multiply(
            current_learning_rate, neighbourhood_function
        )

    @staticmethod
    def calculate_delta(neighborhood_compatible, input_reshaped, weights_batch):
        delta = tf.multiply(
            neighborhood_compatible,
            tf.subtract(
                input_reshaped,
                weights_batch
            )
        )
        return delta

    @staticmethod
    def get_bmu_distance(loc_x, loc_y, batch_size, x, y, input_reshaped, weights_batch):
        bmu = tf.cast(SOM.calculate_bmu(input_reshaped, weights_batch), tf.int32)
        bmu_xs = tf.broadcast_to(
            tf.reshape(tf.math.mod(bmu, x), shape=[batch_size, 1, 1]),
            shape=[batch_size, x, y]
        )
        bmu_ys = tf.broadcast_to(
            tf.reshape(tf.math.floordiv(bmu, x), shape=[batch_size, 1, 1]),
            shape=[batch_size, x, y]
        )
        return tf.cast(
            tf.add(
                tf.pow(tf.subtract(bmu_xs, loc_x), 2),
                tf.pow(tf.subtract(bmu_ys, loc_y), 2)
            ),
            dtype=tf.float32
        )

    def train(self, samples, learning_rate, radius, batch_size=1, global_order=0, qe_callback=False):
        location_x, location_y = self.get_location_matrixes(batch_size)
        samples_size = len(samples)
        batches = []
        for b in list(range(0, samples_size, batch_size)):
            batches.append(samples[b:b + batch_size])
        batch_num = len(batches)
        for current_batch_num in range(batch_num):
            self.weights = self.do_step(
                tf.constant(batches[current_batch_num]),
                tf.constant(batch_size),
                tf.convert_to_tensor(self.weights, dtype=tf.float32),
                location_x,
                location_y,
                tf.constant(self.width),
                tf.constant(self.height),
                tf.constant(self.input_dim),
                tf.constant(current_batch_num),
                tf.constant(batch_num),
                tf.constant(global_order),
                tf.constant(learning_rate),
                tf.constant(radius)
            )
        if qe_callback:
            self.qe = self.calculate_qe(samples)
            logger.info("\rQE= {0:.8f}".format(self.qe / samples.shape[0]))

    def get_bmu_indexes(self, vectors):
        logger.info("\rMapping Input Samples...")
        weights_batch = SOM.broadcast_to_batch(
            self.weights, vectors.shape[0], self.input_dim, self.width, self.height, 0)
        input_reshaped = SOM.broadcast_to_batch(
            vectors, vectors.shape[0], self.input_dim, self.width, self.height, 1)
        t = tf.linalg.norm(
            tf.subtract(
                input_reshaped, weights_batch
            ), axis=2, ord='euclidean'
        )
        indexes = tf.argmin(
            t, axis=0
        ).numpy()
        return indexes

    def get_distances(self, vectors, batch_size=None):

        if batch_size is None:
            batch_size = vectors.shape[0]
        batches = [vectors[i:i + batch_size] for i in range(0, vectors.shape[0], batch_size)]
        distances = None
        for i, batch in enumerate(batches):
            if batch.shape[0] < batch_size:
                batch_size = batch.shape[0]
            weights_batch = SOM.broadcast_to_batch(
                self.weights, batch_size, self.input_dim, self.width, self.height, 0)
            batch_reshaped = SOM.broadcast_to_batch(
                batch, batch_size, self.input_dim, self.width, self.height, 1)
            batch_distances = tf.reduce_sum(
                tf.pow(
                    tf.subtract(weights_batch, batch_reshaped), 2
                ), axis=2
            ).numpy()
            if i > 0:
                distances = np.concatenate((distances, batch_distances), axis=0)
            else:
                distances = batch_distances
        return distances

    def calculate_qe(self, input_vectors):
        distances = self.get_distances(input_vectors)
        return tf.reduce_mean(
            tf.reduce_mean(distances, axis=1),
            axis=0
        )

    def get_location_matrixes(self, batch_size):
        location_x = tf.cast(tf.broadcast_to(
            tf.constant(
                np.array(
                    [
                        [i for i in range(0, self.width)]
                        for _ in range(self.height)
                    ]
                )
            ), [batch_size, self.width, self.height]
        ), dtype=tf.int32)

        location_y = tf.cast(tf.broadcast_to(
            tf.constant(
                np.array(
                    list(itertools.chain.from_iterable(
                        itertools.repeat(j, self.width) for j in range(self.height)
                    ))
                ).reshape(self.width, self.height)
            ), [batch_size, self.width, self.height]
        ), dtype=tf.int32)
        return location_x, location_y

    def get_weights(self):
        return self.weights
