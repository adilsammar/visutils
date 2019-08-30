import os
import tensorflow as tf
import numpy as np


def _load_data(dataset='cifar10'):
    if dataset == 'cifar10':
        # Load CIFAR 10 Data
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        return (x_train, y_train), (x_test, y_test)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_tf_record(data, labels, dataset='cifar10', which='train'):
    """Converts dataset to TFRecords."""
    if dataset == 'cifar10':
        output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataset, '{}'.format(which))
        if not os.path.exists(output_file):
            os.makedirs(output_file)
        print('Creating TFRecords file for {}, at {}'.format(dataset, output_file))
        alias = 1
        for b in np.array_split(range(len(labels)), 2):
            with tf.io.TFRecordWriter(
                    os.path.join(output_file, '{}-{}.tfrecords'.format(which, alias))) as record_writer:
                for i in b:
                    example = tf.train.Example(features=tf.train.Features(
                        feature={
                            'image': _bytes_feature(tf.compat.as_bytes(data[i].tostring())),
                            'label': _int64_feature(labels[i])
                        }))
                    record_writer.write(example.SerializeToString())

            alias += 1


def _generate_tf_records(dataset='cifar10'):
    print('Generating TFRecords for {}'.format(dataset))
    (x_train, y_train), (x_test, y_test) = _load_data(dataset)

    _convert_to_tf_record(x_train, y_train, dataset=dataset, which='train')
    _convert_to_tf_record(x_test, y_test, dataset=dataset, which='test')

    print('Done!')


def get_dataset(dataset='cifar10'):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataset)
    if not os.path.exists(path):
        os.makedirs(path)
        _generate_tf_records(dataset)
    else:
        print('Dataset Exists Reading files')

    return (tf.data.TFRecordDataset(os.path.join(path, 'train.tfrecords')),
            tf.data.TFRecordDataset(os.path.join(path, 'test.tfrecords')))


get_dataset('cifar10')
