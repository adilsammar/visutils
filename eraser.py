# This file defines different random eraser

import tensorflow as tf


def random_erasing(probability=0.1, sl=0.02, sh=0.4, r1=0.3, train_mean=[0.4914009, 0.48215896, 0.4465308]):
    # HWC order
    height = tf.Variable(32, tf.int32)
    width = tf.Variable(32, tf.int32)
    channel = tf.Variable(3, tf.int32)
    area = tf.Variable(1024.0, tf.float32)

    erase_area_low_bound = tf.cast(tf.round(tf.sqrt(sl * area * r1)), tf.int32)
    erase_area_up_bound = tf.cast(tf.round(tf.sqrt((sh * area) / r1)), tf.int32)
    h_upper_bound = tf.minimum(erase_area_up_bound, height)
    w_upper_bound = tf.minimum(erase_area_up_bound, width)

    def eraser(img):
        h = tf.random.uniform([], erase_area_low_bound, h_upper_bound, tf.int32)
        w = tf.random.uniform([], erase_area_low_bound, w_upper_bound, tf.int32)
        x1 = tf.random.uniform([], 0, height + 1 - h, tf.int32)
        y1 = tf.random.uniform([], 0, width + 1 - w, tf.int32)

        # Create mean image
        erase_area = tf.math.subtract(tf.math.add(tf.zeros([h, w, channel]), train_mean),
                                      img[x1:(x1 + h), y1:(y1 + w), :])
        # erase_area = tf.random.uniform([h, w, channel], 0, 1.0, tf.float32) - img[x1:(x1+h), y1:(y1+w), :]
        # Pad patch
        overlay_pad = tf.pad(erase_area, [[x1, height - (x1 + h)], [y1, width - (y1 + w)], [0, 0]])
        # Make final image
        erased_image = img + overlay_pad
        return tf.cond(tf.random.uniform([], 0, 1) > probability, lambda: erased_image, lambda: img)

    return eraser


def random_erasing_x_x(probability=0.5, train_mean=[0.4914009, 0.48215896, 0.4465308], patch_size=(8, 8)):
    # HWC order
    height = tf.Variable(32, tf.int32)
    width = tf.Variable(32, tf.int32)
    channel = tf.Variable(3, tf.int32)
    h = tf.Variable(patch_size[0], tf.int32)
    w = tf.Variable(patch_size[1], tf.int32)

    train_mean = tf.constant(train_mean, tf.float32)

    def eraser(img):
        x1 = tf.random.uniform([], 0, height + 1 - h, tf.int32)
        y1 = tf.random.uniform([], 0, width + 1 - w, tf.int32)

        # Create mean image
        zeros_image = tf.zeros([patch_size[0], patch_size[1], channel], tf.float32)
        mean_image = tf.math.add(zeros_image, train_mean)
        erase_area = tf.math.subtract(mean_image, img[x1:(x1 + h), y1:(y1 + w), :])
        # Pad patch
        overlay_pad = tf.pad(erase_area, [[x1, height - (x1 + h)], [y1, width - (y1 + w)], [0, 0]])
        # Make final image
        erased_image = img + overlay_pad
        return tf.cond(tf.random.uniform([], 0, 1) > probability, lambda: erased_image, lambda: img)

    return eraser
