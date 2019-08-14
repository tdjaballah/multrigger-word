import glob

import numpy as np
from matplotlib import pyplot as plt, mlab as mlab
from pydub import AudioSegment
from scipy.io import wavfile

from src.settings.general import FRAME_RATE, NFFT, RAW_DATA_DIR


def graph_spectrogram(wav_file):

    rate, data = get_wav_info(wav_file)
    nfft = 200      # Length of each window segment
    fs = 8000       # Sampling frequencies
    noverlap = 120  # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    else:
        pxx, freqs, bins, im = plt.specgram(data[:, 0], nfft, fs, noverlap = noverlap)
    return pxx


def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data


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