import logging
import uuid
import random

from src.settings import *
from src.utils import *


def create_one_example(i_label, cut_background, positive, targeted_duration, n_classes):
    y = np.zeros(n_classes)
    y[i_label] = 1
    cut_background = match_target_amplitude(cut_background, -45)
    x = cut_background.overlay(positive,
                               position=np.random.randint(low=0, high=targeted_duration - len(positive)))

    x = np.array(x.get_array_of_samples())
    x = np.swapaxes(get_spectrogram(x), 0, 1)
    print(x.shape)

    return x.reshape(-1), y.reshape(-1)


def create_examples(positives, background, n_examples, targeted_duration, labels):
    labels_to_pick = random.choices(labels, k=n_examples)
    random_picks = np.random.randint(low=0, high=len(background) - targeted_duration, size=n_examples)
    background_picks = [background[r:r + targeted_duration - 1] for r in random_picks]
    result = []

    for (i_label, label), background_pick in zip(labels_to_pick, background_picks):
        positive = random.choice(positives[label])
        result.append(create_one_example(i_label, background_pick, positive, targeted_duration, len(labels)))

    return result


def create_one_tf_record(data_dir, positives, background, targeted_duration, n_samples):
    hashcode = uuid.uuid4()

    result_tf_file = "{}/{}.tfrecord".format(data_dir, hashcode)

    writer = tf.io.TFRecordWriter(result_tf_file)

    labels = list(enumerate(sorted(positives.keys())))

    examples = create_examples(positives, background, n_samples, targeted_duration, labels)

    [writer.write(serialize_example(x, y)) for x, y in examples]

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
