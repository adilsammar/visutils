import os
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()


def _load_data(dataset='cifar10'):
    if dataset == 'cifar10':
        # Load CIFAR 10 Data
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        return (x_train, y_train), (x_test, y_test)


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _serialize_example(image, label, categorical):
    if categorical:
        feature = {
            'image': _bytes_feature(image),
            'label': _bytes_feature(label),
        }
    else:
        feature = {
            'image': _bytes_feature(image),
            'label': _int64_feature(label),
        }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def _convert_to_tf_record(data, labels, dataset='cifar10', which='train', categorical=True):
    """Converts dataset to TFRecords."""
    if dataset == 'cifar10':
        if categorical:
            output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataset, 'categorical',
                                       '{}'.format(which))
        else:
            output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataset, 'direct',
                                       '{}'.format(which))
        if not os.path.exists(output_file):
            os.makedirs(output_file)
        print('Creating TFRecords file for {}, at {}'.format(dataset, output_file))
        alias = 1
        for b in np.array_split(range(len(labels)), len(labels) / 2000):
            with tf.io.TFRecordWriter(
                    os.path.join(output_file, '{}-{}.tfrecords'.format(which, alias))) as record_writer:
                for i in b:
                    if categorical:
                        example = _serialize_example(
                            tf.io.serialize_tensor(data[i].astype('float32') / 255.0),
                            tf.io.serialize_tensor(tf.keras.utils.to_categorical(labels[i].astype('int64'), 10)),
                            categorical
                        )
                    else:
                        example = _serialize_example(
                            tf.io.serialize_tensor(data[i].astype('float32') / 255.0),
                            labels[i].astype('int64'),
                            categorical
                        )
                    record_writer.write(example)

            alias += 1


def _generate_tf_records(dataset='cifar10', categorical=True):
    print('Generating TFRecords for {}'.format(dataset))
    (x_train, y_train), (x_test, y_test) = _load_data(dataset)

    _convert_to_tf_record(x_train, y_train, dataset=dataset, which='train', categorical=categorical)
    _convert_to_tf_record(x_test, y_test, dataset=dataset, which='test', categorical=categorical)

    print('Done!')


def get_dataset(dataset='cifar10', categorical=True):
    if categorical:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataset, 'categorical')
    else:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataset, 'direct')
    if not os.path.exists(path):
        os.makedirs(path)
        _generate_tf_records(dataset, categorical)
    else:
        print('Dataset Already Exists')
