import numpy as np
import random
import tensorflow as tf

from src.utils.audio import cut_audio_segment, get_random_time_segment
from src.settings.general import *


def background_generator(backgrounds, targeted_size, batch_size):
    def gen():
        picked_backgrounds = random.choices(backgrounds, k=batch_size)
        yield [cut_audio_segment(background, targeted_size) for background in picked_backgrounds]
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

        start = int(start * SPECTROGRAM_X / background_length)
        end = int(end * SPECTROGRAM_X / background_length)

        indices = list(enumerate([0] * start + [label] * (end - start) + [0] * (SPECTROGRAM_X - end)))

        Y = tf.SparseTensor(indices=indices, values=[1] * SPECTROGRAM_X,
                            dense_shape=[SPECTROGRAM_X, len(WORDS) + 1])

        return X, tf.sparse.to_dense(Y)

    def fn_batch(backgrounds):
        return tf.map_fn(fn, backgrounds, dtype=(tf.float32, tf.int32))

    return fn_batch


def spectrogram_from_samples_batch(waveforms, labels):
    def spectrogram_from_samples(sample):
        waveform, Y = sample
        signals = tf.reshape(waveform, [1, -1])
        stfts = tf.contrib.signal.stft(signals, frame_length=FFT_FRAME_LENGTH, frame_step=FFT_FRAME_STEP, fft_length=FFT_LENGTH)
        magnitude_spectrograms = tf.abs(stfts)

        num_spectrogram_bins = magnitude_spectrograms.shape[-1].value

        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(NUM_MEL_BINS, num_spectrogram_bins,
                                                                                    FRAME_RATE, LOWER_EDGE_HERTZ,
                                                                                    UPPER_EDGE_HERTZ)

        mel_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
        log_mel_spectrograms = tf.log(mel_spectrograms + tf.keras.backend.epsilon())

        return tf.squeeze(log_mel_spectrograms), Y

    return tf.map_fn(spectrogram_from_samples, (waveforms, labels))


def dataset_input_fn(positives, negatives, backgrounds, negative_ratio, batch_size, num_epochs=None):

    return (tf.data.Dataset.from_generator(background_generator(backgrounds,
                                                                int(BACKGROUND_DURATION_MS*FRAME_RATE/1000),
                                                                batch_size),
                                           output_types=tf.float32)
            .map(add_word(positives, negatives, negative_ratio), num_parallel_calls=N_CORES)
            .map(spectrogram_from_samples_batch, num_parallel_calls=N_CORES)
            .prefetch(PREFETCH_SIZE)
            .repeat(num_epochs)
            )
