import tensorflow as tf


def normalize(mean=[0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
              std=[0.24703225141799082, 0.24348516474564, 0.26158783926049628]):
    return lambda x, y: (tf.math.divide(tf.math.subtract(x, mean), std), y)

