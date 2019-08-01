import logging
import numpy as np
import tensorflow as tf

from src.make_model import seq_model
from src.settings import *


def load_model(weights_dir):
    """
    Load our seq_model with the latest checkpoint
    :param weights_dir: directory where we have our checkpoints from our training
    :return: our sequence model with weights
    """
    latest = tf.train.latest_checkpoint(str(weights_dir))
    model = seq_model(input_shape=(TX, FX),
                      n_classes=N_CLASSES,
                      kernel_size=KERNEL_SIZE,
                      stride=STRIDE)

    return model.load_weights(latest)


def predict(model, x):
    """
    Function to predict the location of one of the trigger word.
    :param x: spectrum of shape (TX, FX)
    :return: numpy array to shape (number of output time steps)
    """

    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)

    return predictions


def get_audio_input_stream(callback):
    stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=1,
        rate=fs,
        input=True,
        frames_per_buffer=chunk_samples,
        input_device_index=0,
        stream_callback=callback)
    return stream


def main():

    model = load_model(CHECKPOINT_DIR)

    return


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
