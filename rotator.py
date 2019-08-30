import tensorflow as tf
import math


def random_rotator(angle=(-10, 10), train_mean=[0.4914009, 0.48215896, 0.4465308]):
    return lambda img: tf.add(
        tf.contrib.image.rotate(
            tf.subtract(img, train_mean),
            tf.random.uniform([], angle[0], angle[1], tf.dtypes.float32) * math.pi / 180
        ),
        train_mean
    )
