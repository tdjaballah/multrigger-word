import numpy as np
import tensorflow as tf


def _dtype_feature(nparray):
    """match appropriate tf.train.Feature class with dtype of ndarray. """
    dtype_ = nparray.dtype
    if dtype_ == np.float64 or dtype_ == np.float32:
        return tf.train.Feature(float_list=tf.train.FloatList(value=nparray))
    elif dtype_ == np.int64:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=nparray))


def trigger_serialize_example(x, y):
    d_feature = {
        'X': _dtype_feature(x),
        'Y': _dtype_feature(y)
    }

    features = tf.train.Features(feature=d_feature)
    example = tf.train.Example(features=features)

    return example.SerializeToString()


def encode_serialize_example(x_1, x_2, y):
    d_feature = {
        'X_1': _dtype_feature(x_1),
        'X_2': _dtype_feature(x_2),
        'Y': _dtype_feature(y)
    }

    features = tf.train.Features(feature=d_feature)
    example = tf.train.Example(features=features)

    return example.SerializeToString()
