import glob
import numpy as np
import os
import tensorflow as tf

from pydub import AudioSegment

from src.settings.general import FRAME_RATE, PROCESSED_DATA_DIR, RAW_DATA_DIR


def normalize_volume():
    """
    Load raw audio into pydub segment to save into wav format the normalized ones
    :return:
    """
    positives_audio_files = glob.glob("{}/positives/*/*.wav".format(RAW_DATA_DIR))
    negatives_audio_files = glob.glob("{}/negatives/*.wav".format(RAW_DATA_DIR))
    background_audio_files = glob.glob("{}/backgrounds/*.wav".format(RAW_DATA_DIR))

    audio_files = positives_audio_files + negatives_audio_files + background_audio_files

    for f in audio_files:
        audio_segment = AudioSegment.from_wav(f)
        audio_segment = match_target_amplitude(audio_segment, -20)
        new_filename = f.replace(str(RAW_DATA_DIR), str(PROCESSED_DATA_DIR))
        os.makedirs(os.path.dirname(new_filename), exist_ok=True)
        audio_segment.export(f.replace(str(RAW_DATA_DIR), str(PROCESSED_DATA_DIR)), format="wav")


def load_processed_audio():
    """
    Load  wav files into numpy array
    :return: numpy array of smaples from pydub audiosegment
    """

    positives, negatives, backgrounds = {}, [], []

    for filepath in glob.glob("{}/positives/*/*.wav".format(PROCESSED_DATA_DIR)):
        label = filepath.split("/")[-2]
        positive = tf.read_file(filepath)
        positive = tf.contrib.ffmpeg.decode_audio(positive, file_format='wav',
                                                  samples_per_second=FRAME_RATE, channel_count=1)
        positive = positive.numpy()
        positives.setdefault(label, [])
        positives[label].append(positive)

    for filepath in glob.glob("{}/negatives/*.wav".format(PROCESSED_DATA_DIR)):
        negative = tf.read_file(filepath)
        negative = tf.contrib.ffmpeg.decode_audio(negative, file_format='wav',
                                                  samples_per_second=FRAME_RATE, channel_count=1)
        negative = negative.numpy()
        negatives.append(negative)

    for filepath in glob.glob("{}/backgrounds/*.wav".format(PROCESSED_DATA_DIR)):
        background = tf.read_file(filepath)
        background = tf.contrib.ffmpeg.decode_audio(background, file_format='wav',
                                                    samples_per_second=FRAME_RATE, channel_count=1)
        background = background.numpy()
        backgrounds.append(background)

    return positives, negatives, backgrounds


def get_random_time_segment(segment_ms, background_duration_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.
    :param segment_ms: the duration of the audio clip in ms ("ms" stands for "milliseconds")
    :param background_duration_ms: the background duration of the audio clip in ms
    :return: tuple of (segment_start, segment_end) in ms
    """

    segment_start = np.random.randint(low=0, high=background_duration_ms - segment_ms)
    segment_end = segment_start + segment_ms

    return segment_start, segment_end


def cut_audio_segment(audio_array, target_size):
    """
    Cut the audio segment to the targeted duration randomly if the audiosegment is shorter than the targeted duration we
    pad it with random silence
    :param audio_array: audio segment to cut in ms
    :param target_size: targeted_duration in ms
    :return: the truncated audio segment
    """
    duration = len(audio_array)

    if target_size < duration:
        segment_start, segment_end = get_random_time_segment(target_size, duration)
        result = audio_array[segment_start:segment_end]

    else:
        segment_start, segment_end = get_random_time_segment(duration, target_size)

        result = np.pad(audio_array, (segment_start, duration - segment_end))

    assert len(result) == target_size

    return result


def match_target_amplitude(sound, target_dBFS):
    """
    Used to standardize volume of audio clip
    :param sound: sound to standardize
    :param target_dBFS: targeted volume
    :return: standardized sound
    """
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)
