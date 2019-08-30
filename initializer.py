import tensorflow as tf


def init_kernal(shape, dtype=tf.float32, partition_info=None):
    fan = tf.math.reduce_prod(shape[:-1])
    bound = 1 / tf.math.sqrt(fan)
    return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)
