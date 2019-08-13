import logging
import uuid
import random

from src.settings import *
from src.utils import *


def create_one_sound(cut_background, positive):

    cut_background = match_target_amplitude(cut_background, -45)
    x = cut_background.overlay(positive,
                               position=np.random.randint(low=0, high=len(cut_background) - len(positive)))

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

    for (label_1, label_2), (background_1, background_2) in zip(labels_to_pick, background_picks):

        sound_1, sound_2 = random.choice(positives[label_1]), random.choice(positives[label_2])

        sound_1 = create_one_sound(background_1, sound_1)
        sound_2 = create_one_sound(background_2, sound_2)

        if label_1 == label_2:
            target = np.ones(1)
        else:
            target = np.zeros(1)

        result.append((sound_1, sound_2, target))

    return result


def create_one_tf_record(data_dir, positives, background, targeted_duration, n_samples):
    hashcode = uuid.uuid4()

    result_tf_file = "{}/{}.tfrecord".format(data_dir, hashcode)

    writer = tf.io.TFRecordWriter(result_tf_file)

    labels = sorted(positives.keys())

    examples = create_examples(positives, background, n_samples, targeted_duration, labels)

    [writer.write(classif_serialize_example(x_1, x_2, y)) for x_1, x_2, y in examples]

    writer.close()


def main(n_dev_samples, n_val_samples):

    positives, backgrounds = load_raw_audio()
    background = backgrounds[0]

    files_to_delete = glob.glob("{}/*/*.tfrecord".format(CLASSIF_PROCESSED_DATA_DIR))
    clean_data_dir(files_to_delete)

    targeted_duration = 2000

    for i in range(n_dev_samples // N_SAMPLES_IN_TFRECORD):
        create_one_tf_record(DEV_CLASSIF_PROCESSED_DATA_DIR, positives, background, targeted_duration, N_SAMPLES_IN_TFRECORD)
        print("dev", i)

    for i in range(n_val_samples // N_SAMPLES_IN_TFRECORD):
        create_one_tf_record(VAL_CLASSIF_PROCESSED_DATA_DIR, positives, background, targeted_duration, N_SAMPLES_IN_TFRECORD)
        print("val", i)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main(n_dev_samples=N_CLASSIF_DEV_SAMPLES, n_val_samples=N_CLASSIF_VAL_SAMPLES)
