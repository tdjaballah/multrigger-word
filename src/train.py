import logging
import tensorflow as tf

from src.make_model import seq_model
from src.settings import *
from src.utils import f1_m, precision_m, recall_m


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
    #dataset = dataset.shuffle(buffer_size=N_SAMPLES)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(5)
    dataset = dataset.repeat(num_epochs)

    return dataset


def main(n_epochs=10, batch_size=64):

    train_steps_per_epoch = int(N_DEV_SAMPLES / batch_size)
    val_steps_per_epoch = int(N_VAL_SAMPLES / batch_size)

    dev_tfrecord_files = glob.glob("{}/*.tfrecord".format(DEV_PROCESSED_DATA_DIR))
    training_set = dataset_input_fn(dev_tfrecord_files, batch_size)

    val_tfrecord_files = glob.glob("{}/*.tfrecord".format(VAL_PROCESSED_DATA_DIR))
    validation_set = dataset_input_fn(val_tfrecord_files, batch_size)

    model = seq_model(input_shape=(TX, FX),
                      n_classes=N_CLASSES,
                      kernel_size=KERNEL_SIZE,
                      stride=STRIDE)

    opt = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy", f1_m, precision_m, recall_m])

    csv_logger = tf.keras.callbacks.CSVLogger(str(TRAIN_LOG_FILE))

    cp_callback = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_FILES,
                                                     verbose=1,
                                                     save_weights_only=True,
                                                     period=5)

    model.fit(training_set.make_one_shot_iterator(),
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
    main()
