import glob
import logging
import tensorflow as tf

from src.settings import *
from src.make_model import seq_model


def _extract_feature(record, feature):
    example = tf.train.Example.FromString(record.numpy())
    return example.features.feature[feature].float_list.value


# Load tf record dataset
def parser(record):
    """
    parse tf record
    :param record: the tf record to parse
    :return: tensor
    """
    print(type(TX), type(FX), type(TY), type(N_CLASSES))
    X = tf.reshape(
        tf.py_function(
            lambda r: _extract_feature(r, "X"),
            (record,),
            tf.float32
        ), [TX, FX]
    )

    Y = tf.reshape(
        tf.py_function(
            lambda r: _extract_feature(r, "Y"),
            (record,),
            tf.float32
        ), [TY, N_CLASSES]
    )

    return X, Y


def dataset_input_fn(filenames, batch_size, num_epochs=None):
    """
    the input function we use to feed our keras model
    :param filenames: tfrecords filenames
    :param batch_size: size of the train size
    :param num_epochs: num_epochs
    :return: tf.Dataset
    """
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(5)
    dataset = dataset.repeat(num_epochs)
    # iterator = dataset.make_one_shot_iterator()
    # features, labels = iterator.get_next()

    return dataset


def main(epochs=100, steps_per_epoch=1000, batch_size=16):

    print(PROJECT_DIR)
    print(DATA_DIR)
    print(PROCESSED_DATA_DIR)

    tfrecord_files = glob.glob("{}/*.tfrecord".format(PROCESSED_DATA_DIR))

    training_set = dataset_input_fn(tfrecord_files, batch_size, None)

    model = seq_model(input_shape=(TX, FX),
                      n_classes=N_CLASSES,
                      kernel_size=KERNEL_SIZE,
                      stride=STRIDE)

    opt = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

    csv_logger = tf.keras.callbacks.CSVLogger(TRAIN_LOG_FILE)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_FILES,
                                                     verbose=1,
                                                     save_weights_only=True,
                                                     period=5)

    model.fit(training_set.make_one_shot_iterator(),
              steps_per_epoch=steps_per_epoch,
              epochs=epochs,
              callbacks=[csv_logger, cp_callback],
              verbose=1
              )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
