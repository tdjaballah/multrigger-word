import logging
import uuid
import random
import tensorflow as tf
import numpy as np
import glob

from src.utils.audio import match_target_amplitude, get_spectrogram, load_raw_audio
from src.utils.misc_utils import clean_data_dir
from src.utils.tf_record import encode_serialize_example

from src.settings.encode import *


def create_one_sound(cut_background, positive, export=None):

    cut_background = match_target_amplitude(cut_background, -45)
    x = cut_background.overlay(positive,
                               position=np.random.randint(low=0, high=len(cut_background) - len(positive)))

    if export:
        x.export("{}/{}.wav".format(ENCODE_INTERIM_DATA_DIR, export), format="wav")

    x = np.array(x.get_array_of_samples())
    x = np.swapaxes(get_spectrogram(x), 0, 1)

    return x.reshape(-1)


def create_examples(positives, background, n_examples, targeted_duration, labels):

    labels_to_pick = random.choices(labels, k=n_examples)

    labels_to_pick = [(label, random.choice([label, random.choice([l for l in labels if l != label])])) for label in labels_to_pick]

    random_picks = np.random.randint(low=0, high=len(background) - targeted_duration, size=(n_examples, 2))
    background_picks = [(background[r_1:r_1 + targeted_duration - 1], background[r_2:r_2 + targeted_duration - 1])
                        for r_1, r_2 in random_picks]
    result = []

    i = 1

    for (label_1, label_2), (background_1, background_2) in zip(labels_to_pick, background_picks):

        if label_1 == label_2:
            target = np.ones(1)
        else:
            target = np.zeros(1)

        if i == n_examples:
            export = "{}_{}".format(label_1 == label_2, uuid.uuid4())
            export_1, export_2 = export + "_1", export + "_2"
        else:
            export_1, export_2 = None, None

        sound_1, sound_2 = random.choice(positives[label_1]), random.choice(positives[label_2])

        sound_1 = create_one_sound(background_1, sound_1, export_1)
        sound_2 = create_one_sound(background_2, sound_2, export_2)

        result.append((sound_1, sound_2, target))

        i += 1

    return result


def create_one_tf_record(data_dir, positives, background, targeted_duration, n_samples):
    hashcode = uuid.uuid4()

    result_tf_file = "{}/{}.tfrecord".format(data_dir, hashcode)

    writer = tf.io.TFRecordWriter(result_tf_file)

    labels = sorted(positives.keys())

    examples = create_examples(positives, background, n_samples, targeted_duration, labels)

    [writer.write(encode_serialize_example(x_1, x_2, y)) for x_1, x_2, y in examples]

    writer.close()


def main(n_dev_samples, n_val_samples):

    positives, backgrounds = load_raw_audio()
    background = backgrounds[0]

    files_to_delete = glob.glob("{}/*/*.tfrecord".format(ENCODE_PROCESSED_DATA_DIR))
    sample_files = glob.glob("{}/*".format(ENCODE_INTERIM_DATA_DIR))
    files_to_delete.extend(sample_files)

    clean_data_dir(files_to_delete)

    targeted_duration = 2000

    for i in range(n_dev_samples // N_SAMPLES_IN_TFRECORD):
        create_one_tf_record(DEV_ENCODE_PROCESSED_DATA_DIR, positives, background, targeted_duration, N_SAMPLES_IN_TFRECORD)
        print("dev", i)

    for i in range(n_val_samples // N_SAMPLES_IN_TFRECORD):
        create_one_tf_record(VAL_ENCODE_PROCESSED_DATA_DIR, positives, background, targeted_duration, N_SAMPLES_IN_TFRECORD)
        print("val", i)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main(n_dev_samples=N_ENCODE_DEV_SAMPLES, n_val_samples=N_ENCODE_VAL_SAMPLES)
