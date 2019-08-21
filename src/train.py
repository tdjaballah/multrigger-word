import logging
import numpy as np
import random
import tensorflow as tf

from src.utils.audio import load_processed_audio, cut_audio_segment, get_random_time_segment
from src.settings.general import *


def background_generator(backgrounds, targeted_size):
    def gen():
        background = random.choice(backgrounds)
        yield cut_audio_segment(background, targeted_size)
    return gen


def add_word(positives, negatives, negative_ratio):
    """
    Add words to background record
    :param words: word
    :return: new background with the word add randomly in it
    """

    def fn(background):

        if np.random.random() > negative_ratio:

            label, word_label_to_add = random.choice(list(enumerate(WORDS, 1)))
            word_to_add = random.choice(positives[word_label_to_add])

        else:
            word_to_add, label = random.choice(negatives), 0

        word_length = len(word_to_add)
        background_length = int((BACKGROUND_DURATION_MS / 1000) * FRAME_RATE)

        start, end = get_random_time_segment(word_length, background_length)

        word_to_add = tf.convert_to_tensor(np.array(word_to_add), dtype=tf.float32)

        X = tf.concat([background[:start], word_to_add, background[end:]], axis=0)

        indices = list(enumerate([0] * start + [label] * (end - start) + [0] * (background_length - end)))

        Y = tf.SparseTensor(indices=indices, values=[1] * background_length,
                            dense_shape=[background_length, len(WORDS) + 1])

        return X, tf.sparse.to_dense(Y)

    return fn


def spectrogram_from_samples(waveform, Y):
    signals = tf.reshape(waveform, [1, -1])
    stfts = tf.contrib.signal.stft(signals, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP, fft_length=FFT_LENGTH)
    magnitude_spectrograms = tf.abs(stfts)

    num_spectrogram_bins = magnitude_spectrograms.shape[-1].value

    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(NUM_MEL_BINS, num_spectrogram_bins,
                                                                                FRAME_RATE, LOWER_EDGE_HERTZ,
                                                                                UPPER_EDGE_HERTZ)

    mel_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
    log_mel_spectrograms = tf.log(mel_spectrograms + tf.keras.backend.epsilon())

    return log_mel_spectrograms, Y


def dataset_input_fn(backgrounds, positives, negatives, negative_ratio): #, batch_size, words, num_epochs=None):
    """
    the input function we use to feed our keras model
    :param background_tf_records: tfrecords filenames
    :param batch_size: size of the train size
    :param num_epochs: num_epochs
    :return: tf.Dataset
    """

    return (tf.data.Dataset.from_generator(background_generator(backgrounds, BACKGROUND_DURATION_MS*FRAME_RATE))
            .map(add_word(positives, negatives, negative_ratio))
            .map(spectrogram_from_samples)
            )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    positives, negatives, backgrounds = load_processed_audio()