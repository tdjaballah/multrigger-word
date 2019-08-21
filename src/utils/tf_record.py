import numpy as np
import tensorflow as tf


def _dtype_feature(nparray):
    """match appropriate tf.train.Feature class with dtype of ndarray. """
    dtype_ = nparray.dtype

    if dtype_ in [np.float64,  np.float32]:
        return tf.train.Feature(float_list=tf.train.FloatList(value=nparray))

    elif dtype_ in [np.int64, np.int32, np.int16]:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=nparray))


def serialize_example(x):
    d_feature = {
        'audio': _dtype_feature(x)
    }

    features = tf.train.Features(feature=d_feature)
    example = tf.train.Example(features=features)

    return example.SerializeToString()


def _parse_function(record):
    """Extracts features and labels.

    Args:
      record: File path to a TFRecord file
    Returns:
      A `tuple` `(labels, features)`:
        features: A dict of tensors representing the features
        labels: A tensor with the corresponding labels.
    """
    features = {
        "audio": tf.io.VarLenFeature(dtype=tf.int64)  # terms are strings of varying lengths
    }

    parsed_features = tf.parse_single_example(record, features)

    return tf.sparse.to_dense(parsed_features['audio'])
