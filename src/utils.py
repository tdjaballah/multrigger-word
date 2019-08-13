import glob
import matplotlib.pyplot as plt
import numpy as np
import os

import tensorflow as tf

from matplotlib import mlab as mlab
from pydub import AudioSegment
from scipy.io import wavfile

from src.settings import RAW_DATA_DIR, FRAME_RATE, NFFT


def clean_data_dir(files):
    [os.remove(file) for file in files]


# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file):

    rate, data = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    else:
        pxx, freqs, bins, im = plt.specgram(data[:, 0], nfft, fs, noverlap = noverlap)
    return pxx


# Load a wav file
def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data


def f1_scores_1(y_true, y_pred):
    """Computes 3 different f1 scores (micro, macro, weighted).
    micro: f1-score based on overall precision and recall
    macro: average f1-score on all classes
    weighted: weighted average of f1-scores on all classes, using the number of supporting observations of each class
    Args:
        y_true (Tensor): predictions, same shape as y
        y_pred (Tensor): labels, with shape (batch_size, num_classes)
        thresh: probability value beyond which we predict positive
    Returns:
        tuple(Tensor): (micro, macro, weighted) tuple of the computed f1 scores
    """
    f1s = [0, 0, 0]

    y_pred = tf.cast(tf.round(y_pred), tf.float32)
    for i, axis in enumerate([None, 0]):
        TP = tf.cast(tf.count_nonzero(y_pred * y_true, axis=axis), tf.float32)
        FP = tf.cast(tf.count_nonzero(y_pred * (1 - y_true), axis=axis), tf.float32)
        FN = tf.cast(tf.count_nonzero((1 - y_pred) * y_true, axis=axis), tf.float32)
        precision = TP / (TP + FP + 1e-16)
        recall = TP / (TP + FN + 1e-16)
        f1 = 2 * precision * recall / (precision + recall + 1e-16)
        f1s[i] = tf.reduce_mean(f1)
    weights = tf.reduce_sum(y_pred, axis=0)
    weights /= tf.reduce_sum(weights)
    f1s[2] = tf.reduce_sum(f1 * weights)
    return f1s[0]


def f1_scores_2(y_true, y_pred):
    """Computes 3 different f1 scores (micro, macro, weighted).
    micro: f1-score based on overall precision and recall
    macro: average f1-score on all classes
    weighted: weighted average of f1-scores on all classes, using the number of supporting observations of each class
    Args:
        y_true (Tensor): predictions, same shape as y
        y_pred (Tensor): labels, with shape (batch_size, num_classes)
        thresh: probability value beyond which we predict positive
    Returns:
        tuple(Tensor): (micro, macro, weighted) tuple of the computed f1 scores
    """
    f1s = [0, 0, 0]

    y_pred = tf.cast(tf.round(y_pred), tf.float32)
    for i, axis in enumerate([None, 0]):
        TP = tf.cast(tf.count_nonzero(y_pred * y_true, axis=axis), tf.float32)
        FP = tf.cast(tf.count_nonzero(y_pred * (1 - y_true), axis=axis), tf.float32)
        FN = tf.cast(tf.count_nonzero((1 - y_pred) * y_true, axis=axis), tf.float32)
        precision = TP / (TP + FP + 1e-16)
        recall = TP / (TP + FN + 1e-16)
        f1 = 2 * precision * recall / (precision + recall + 1e-16)
        f1s[i] = tf.reduce_mean(f1)
    weights = tf.reduce_sum(y_pred, axis=0)
    weights /= tf.reduce_sum(weights)
    f1s[2] = tf.reduce_sum(f1 * weights)
    return f1s[1]


def f1_scores_3(y_true, y_pred):
    """Computes 3 different f1 scores (micro, macro, weighted).
    micro: f1-score based on overall precision and recall
    macro: average f1-score on all classes
    weighted: weighted average of f1-scores on all classes, using the number of supporting observations of each class
    Args:
        y_true (Tensor): predictions, same shape as y
        y_pred (Tensor): labels, with shape (batch_size, num_classes)
    Returns:
        tuple(Tensor): (micro, macro, weighted) tuple of the computed f1 scores
    """
    f1s = [0, 0, 0]

    y_pred = tf.cast(tf.round(y_pred), tf.float32)
    for i, axis in enumerate([None, 0]):
        TP = tf.cast(tf.count_nonzero(y_pred * y_true, axis=axis), tf.float32)
        FP = tf.cast(tf.count_nonzero(y_pred * (1 - y_true), axis=axis), tf.float32)
        FN = tf.cast(tf.count_nonzero((1 - y_pred) * y_true, axis=axis), tf.float32)
        precision = TP / (TP + FP + 1e-16)
        recall = TP / (TP + FN + 1e-16)
        f1 = 2 * precision * recall / (precision + recall + 1e-16)
        f1s[i] = tf.reduce_mean(f1)
    weights = tf.reduce_sum(y_pred, axis=0)
    weights /= tf.reduce_sum(weights)
    f1s[2] = tf.reduce_sum(f1 * weights)
    return f1s[2]


def _soft_f1_macro(y_hat, y):
    """Computes the soft macro f1-score (average f1-score when we consider probability predictions for each class)
    Args:
        y_hat (Tensor): predictions, same shape as y
        y (Tensor): labels, with shape (batch_size, num_classes)
    Returns:
        tuple(Tensor): (micro, macro, weighted) tuple of the computed f1 scores
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    TP = tf.reduce_sum(y_hat * y, axis=0)
    FP = tf.reduce_sum(y_hat * (1 - y), axis=0)
    FN = tf.reduce_sum((1 - y_hat) * y, axis=0)
    precision = TP / (TP + FP + 1e-16)
    recall = TP / (TP + FN + 1e-16)
    f1 = 2 * precision * recall / (precision + recall + 1e-16)
    # reduce 1-f1 in order to increase f1
    soft_f1 = 1 - f1
    soft_f1 = tf.reduce_mean(soft_f1)
    return soft_f1


def load_raw_audio():
    positives = {}
    backgrounds = []

    for filepath in glob.glob("{}/positives/*/*.wav".format(RAW_DATA_DIR)):
        label = filepath.split("/")[-2]
        positive = AudioSegment.from_wav(filepath).set_frame_rate(FRAME_RATE).set_channels(1)
        positive = match_target_amplitude(positive, -20.0)
        positives.setdefault(label, [])
        positives[label].append(positive)

    for filepath in glob.glob("{}/backgrounds/*.wav".format(RAW_DATA_DIR)):
        background = AudioSegment.from_wav(filepath).set_frame_rate(FRAME_RATE).set_channels(1)
        background = match_target_amplitude(background, -20.0)
        backgrounds.append(background)

    return positives, backgrounds


def get_random_time_segment(segment_ms, background_duration_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.
    :param segment_ms: the duration of the audio clip in ms ("ms" stands for "milliseconds")
    :param background_duration_ms: the background duration of the audio clip in ms
    :return: tuple of (segment_start, segment_end) in ms
    """

    segment_start = np.random.randint(low=0, high=background_duration_ms - segment_ms)
    segment_end = segment_start + segment_ms - 1

    return segment_start, segment_end


def cut_audio_segment(audio_segment, targeted_duration):
    """
    Cut the audio segment to the targeted duration randomly
    :param audio_segment: audio segment to cut
    :param targeted_duration: targeted_duration
    :return: the truncated audio segment
    """
    duration = len(audio_segment)
    if targeted_duration < duration:
        segment_start = np.random.randint(low=0, high=duration-targeted_duration)
        segment_end = segment_start + targeted_duration - 1
        return audio_segment[segment_start:segment_end]
    else:
        return audio_segment


def match_target_amplitude(sound, target_dBFS):
    """
    Used to standardize volume of audio clip
    :param sound: sound to standardize
    :param target_dBFS: targeted volume
    :return: standardized sound
    """
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def get_spectrogram(data, fs=2):
    """
    Get spectrogram from raw audio data
    :param data: raw audio data
    :param fs:
    :return:
    """

    nchannels = data.ndim

    if nchannels > 1:
        data = data[:, 0]

    pxx, _, _ = mlab.specgram(data, NFFT, fs, noverlap=int(NFFT / 2))

    return pxx


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


def classif_serialize_example(x_1, x_2, y):
    d_feature = {
        'X_1': _dtype_feature(x_1),
        'X_2': _dtype_feature(x_2),
        'Y': _dtype_feature(y)
    }

    features = tf.train.Features(feature=d_feature)
    example = tf.train.Example(features=features)

    return example.SerializeToString()


def _extract_feature(record, feature):
    example = tf.train.Example.FromString(record.numpy())
    return example.features.feature[feature].float_list.value