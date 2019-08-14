import logging
import tensorflow as tf
import glob

from src.settings.general import N_CORES
from src.settings.encode import *
from src.models import siamese_model


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
        "X_1": tf.FixedLenFeature(shape=[343, 257], dtype=tf.float32),  # terms are strings of fixed lengths
        "X_2": tf.FixedLenFeature(shape=[343, 257], dtype=tf.float32),
        "Y": tf.FixedLenFeature(shape=[1], dtype=tf.float32)
    }

    parsed_features = tf.parse_single_example(record, features)

    X_1 = parsed_features['X_1']
    X_2 = parsed_features['X_2']
    Y = parsed_features['Y']

    return (X_1, X_2), Y


def encode_dataset_input_fn(filenames, batch_size, num_epochs=None):
    """
    the input function we use to feed our keras model
    :param filenames: tfrecords filenames
    :param batch_size: size of the train size
    :param num_epochs: num_epochs
    :return: tf.Dataset
    """
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=N_CORES)
    dataset = dataset.map(_parse_function, num_parallel_calls=N_CORES)
    dataset = dataset.shuffle(buffer_size=int(5*batch_size)+1)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(5)
    dataset = dataset.repeat(num_epochs)

    return dataset


def main(n_epochs, batch_size):

    train_steps_per_epoch = int(N_ENCODE_DEV_SAMPLES / batch_size)
    val_steps_per_epoch = int(N_ENCODE_VAL_SAMPLES / batch_size)

    dev_tfrecord_files = glob.glob("{}/*.tfrecord".format(DEV_ENCODE_PROCESSED_DATA_DIR))
    training_set = encode_dataset_input_fn(dev_tfrecord_files, batch_size)

    val_tfrecord_files = glob.glob("{}/*.tfrecord".format(VAL_ENCODE_PROCESSED_DATA_DIR))
    validation_set = encode_dataset_input_fn(val_tfrecord_files, batch_size)

    one_shot_model = siamese_model(input_shape=(343, 257),
                                   kernel_size=ENCODE_KERNEL_SIZE)

    opt = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)

    one_shot_model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    csv_logger = tf.keras.callbacks.CSVLogger(str(ENCODE_TRAINING_LOG_FILE))

    cp_callback = tf.keras.callbacks.ModelCheckpoint(ENCODE_CHECKPOINT_FILES,
                                                     verbose=1,
                                                     save_weights_only=True,
                                                     period=5)

    one_shot_model.fit(training_set.make_one_shot_iterator(),
                       validation_data=validation_set.make_one_shot_iterator(),
                       validation_steps=val_steps_per_epoch,
                       steps_per_epoch=train_steps_per_epoch,
                       epochs=n_epochs,
                       callbacks=[csv_logger, cp_callback],
                       verbose=1
                       )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main(ENCODE_EPOCHS, ENCODE_BATCH_SIZE)
