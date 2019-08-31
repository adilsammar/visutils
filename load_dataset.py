import os
import tensorflow as tf
import glob


def _parse_tf_record(serialized_example):
    feature_description = {
        'image': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.int64)
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.parse_tensor(example['image'], out_type=float)

    return image, example['label']


def get_dataset(dataset='cifar10'):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataset)
    if not os.path.exists(path):
        raise Exception("Dataset Not Found with name {}".format(dataset))
    else:
        print('Dataset Exists Reading files')

    return (
        tf.data.TFRecordDataset([f for f in glob.glob(os.path.join(path, 'train', "*.tfrecords"))]).map(
            _parse_tf_record),
        tf.data.TFRecordDataset([f for f in glob.glob(os.path.join(path, 'test', "*.tfrecords"))]).map(
            _parse_tf_record)
    )
