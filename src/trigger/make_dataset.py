import logging
import pandas as pd
import random
import seaborn as sns
import numpy as np
import uuid
import glob
import tensorflow as tf

from src.settings.trigger import *
from src.utils.audio import load_raw_audio, get_random_time_segment, cut_audio_segment, get_spectrogram
from src.utils.tf_record import trigger_serialize_example
from src.utils.misc_utils import clean_data_dir


def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.
    :param segment_time: a tuple of (segment_start, segment_end) for the new segment
    :param previous_segments: a list of tuples of (segment_start, segment_end) for the existing segments
    :return: True if the time segment overlaps with any of the existing segments, False otherwise
    """

    segment_start, segment_end = segment_time

    # Step 1: Initialize overlap as a "False" flag. (≈ 1 line)
    overlap = False

    # Step 2: loop over the previous_segments start and end times.
    for previous_start, previous_end in previous_segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True

    return overlap


def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the
    audio segment does not overlap with existing segments.
    :param background: background audio recording
    :param audio_clip: audio clip to be inserted/overlaid
    :param previous_segments: times where audio segments have already been placed
    :return: the updated background audio
    """

    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)
    # Get the duration of the background clip in ms
    background_duration_ms = len(background)

    segment_time = get_random_time_segment(segment_ms, background_duration_ms)

    i = 0
    while is_overlapping(segment_time, previous_segments):
        if i < 10:
            segment_time = get_random_time_segment(segment_ms, background_duration_ms)
            i += 1
        else:
            return background, (background_duration_ms, background_duration_ms)

    previous_segments.append(segment_time)

    new_background = background[:segment_time[0]+CROSSFADE_MS].append(audio_clip, crossfade=CROSSFADE_MS)
    new_background = new_background.append(background[segment_time[1]-CROSSFADE_MS:], crossfade=CROSSFADE_MS)

    return new_background, segment_time


def insert_ones(y, y_label, segment_end_ms, background_duration_ms, label_duration):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 followinf labels should be ones.
    :param y: numpy array of shape (1, Ty), the labels of the training example
    :pram y_label: number otf the class we have to put 1
    :param segment_end_ms: the end time of the segment in ms
    :param background_duration_ms: duration of the background in ms
    :param label_duration: duration of the label
    :return:
    """

    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * TY / background_duration_ms)

    # Add 1 to the correct index in the background label (y)
    for i in range(segment_end_y, segment_end_y + label_duration):
        if i < TY:
            y[i, y_label] = 1
            y[i, 0] = 0

    return y


def transform_labels(y, map_dict):
    """
    Save figure in sample dir to visualize our generated labels
    :param y: ndarray of shape (TY, N_CLASSES)
    :return: save file png
    """
    df = pd.DataFrame(y)
    df = pd.concat([pd.DataFrame({'label': i, 'x': df.index, 'y': list(df[i])}) for i in df.columns])
    df['trigger'] = df['label'] != 0
    df['label'] = df['label'].map(map_dict)

    return df


def create_training_example(background, background_duration_ms, label_duration, positives, positive_labels, type_set, export, hashcode):
    """
    Creates a training example with a given background, activates, and negatives.

    :param background:   background audio recording
    :param background_duration_ms: background duration we want in ms
    :param label_duration: number of ones to add in the label
    :param positives: list of audio segments of the positives word we want to detect
    :param export:
    :return: tuple (x,y) with
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """

    map_dict = dict(enumerate(positive_labels, 1))
    map_dict[0] = "background"

    background = cut_audio_segment(background, background_duration_ms)

    # Step 1: Initialize y (label vector) of zeros (≈ 1 line)
    y = np.zeros((TY, N_WORDS + 1))
    y[:, 0] = 1

    # Step 2: Initialize segment times as empty list (≈ 1 line)
    previous_segments = []

    # Select 0-2 random "activate" audio clips from the entire list of "activates" recordings for 10 seconds record
    number_of_sound_to_add = np.random.randint(3 * background_duration_ms / 5000)

    for i in range(number_of_sound_to_add):

        # Take random positive with random label and get the right amplitude
        y_label, label = random.choice(list(enumerate(positive_labels, 1)))
        random_positive = random.choice(positives[label])

        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_positive, previous_segments)

        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time

        # Insert labels in "y"
        y = insert_ones(y, y_label, segment_end, background_duration_ms, label_duration)

    if export:

        # Export new training example
        file_name = "{}/{}-{}".format(TRIGGER_INTERIM_DATA_DIR, type_set, hashcode)
        background.export(file_name + ".wav", format="wav")

        #Export label as a graph
        df_y = transform_labels(y, map_dict)
        sns.relplot(x='x', y='y', row='trigger', hue='label', data=df_y, kind='line', legend="full").savefig(file_name + ".png")

    background = np.array(background.get_array_of_samples())
    x = np.swapaxes(get_spectrogram(background), 0, 1)

    return x.reshape(-1), y.reshape(-1)


def create_one_tf_record(data_dir, sample_duration_ms, label_duration, positives, positive_labels, background, type_set):
    """
    Create one tfrecord with 50 msplae in it.
    :param data_dir: data dir to write tfrecord
    :param sample_duration_ms: smaple duration in milliseconds
    :param label_duration: number of ones to put in label
    :param raw_audio: result from load_audio_files
    :param type_set: 'dev' or 'val' values allowed
    :return:
    """

    hashcode = uuid.uuid4()

    result_tf_file = "{}/{}.tfrecord".format(data_dir, hashcode)

    writer = tf.io.TFRecordWriter(result_tf_file)

    examples = []

    for i in range(N_SAMPLES_IN_TFRECORD):
        if i == N_SAMPLES_IN_TFRECORD - 1:
            export = True
        else:
            export = False

        examples.append(create_training_example(background, sample_duration_ms,
                                                label_duration, positives, positive_labels,
                                                type_set, export, hashcode))

    [writer.write(trigger_serialize_example(x, y)) for x, y in examples]

    writer.close()


def main(n_dev_samples, n_val_samples):

    """
    Runs data processing scripts to turn raw audio data from (../raw/*) into
    cleaned tfrecord data ready to be analyzed (saved in ../processed) by trigger word neural network.
    :param n_dev_samples: number of training samples
    :param n_val_samples: number of validation samples
    :return: tfrecords in specified directories
    """

    files_to_delete = glob.glob("{}/*/*.tfrecord".format(TRIGGER_PROCESSED_DATA_DIR))
    sample_files = glob.glob("{}/*".format(TRIGGER_INTERIM_DATA_DIR))
    files_to_delete.extend(sample_files)

    clean_data_dir(files_to_delete)

    positives, backgrounds = load_raw_audio()
    positive_labels = sorted(positives.keys())
    background = backgrounds[0]

    for i in range(n_dev_samples // N_SAMPLES_IN_TFRECORD):
        create_one_tf_record(DEV_TRIGGER_PROCESSED_DATA_DIR, SAMPLE_DURATION_MS, LABEL_DURATION, positives, positive_labels, background, "dev")
        print("dev", i)

    for i in range(n_val_samples // N_SAMPLES_IN_TFRECORD):
        create_one_tf_record(VAL_TRIGGER_PROCESSED_DATA_DIR, SAMPLE_DURATION_MS, LABEL_DURATION, positives, positive_labels, background, "val")
        print("val", i)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main(n_dev_samples=N_TRIGGER_DEV_SAMPLES, n_val_samples=N_TRIGGER_VAL_SAMPLES)
