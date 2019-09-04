import os
import tensorflow as tf
import glob
from . import dataset as ds


def _parse_tf_record(categorical):
    if categorical:
        feature_description = {
            'image': tf.io.FixedLenFeature((), tf.string),
            'label': tf.io.FixedLenFeature((), tf.string)
        }
    else:
        feature_description = {
            'image': tf.io.FixedLenFeature((), tf.string),
            'label': tf.io.FixedLenFeature((), tf.int64)
        }

    def _parse(serialized_example):
        example = tf.io.parse_single_example(serialized_example, feature_description)
        image = tf.io.parse_tensor(example['image'], out_type=tf.float32)
        if categorical:
            label = tf.io.parse_tensor(example['label'], out_type=tf.float32)
        else:
            label = example['label']

        return image, label

    return _parse


def get_dataset(dataset='cifar10', categorical=True):
    if categorical:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataset, 'categorical')
    else:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataset, 'direct')
    if not os.path.exists(path):
        ds.get_dataset(dataset, categorical)
    else:
        print('Dataset Exists Reading files')

    _parse = _parse_tf_record(categorical)

    return (
        tf.data.TFRecordDataset([f for f in glob.glob(os.path.join(path, 'train', '*.tfrecords'))]).map(
            _parse),
        tf.data.TFRecordDataset([f for f in glob.glob(os.path.join(path, 'test', '*.tfrecords'))]).map(
            _parse)
    )