import matplotlib.pyplot as plt
import os
import tensorflow as tf

from pydub import AudioSegment
from scipy.io import wavfile


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


# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


# Load raw audio files for speech synthesis
def load_raw_audio():
    activates = []
    backgrounds = []
    negatives = []
    for filename in os.listdir("./raw_data/activates"):
        if filename.endswith("wav"):
            activate = AudioSegment.from_wav("./raw_data/activates/"+filename)
            activates.append(activate)
    for filename in os.listdir("./raw_data/backgrounds"):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav("./raw_data/backgrounds/"+filename)
            backgrounds.append(background)
    for filename in os.listdir("./raw_data/negatives"):
        if filename.endswith("wav"):
            negative = AudioSegment.from_wav("./raw_data/negatives/"+filename)
            negatives.append(negative)
    return activates, negatives, backgrounds


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