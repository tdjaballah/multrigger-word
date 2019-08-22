import logging
import tensorflow as tf

from src.utils.tf_helper import _soft_f1_macro, f1_scores_1, f1_scores_2, f1_scores_3
from src.utils.audio import load_processed_audio
from src.make_dataset import dataset_input_fn
from src.models import nn
from src.settings.general import *

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    tf.enable_eager_execution()

    positives, negatives, backgrounds = load_processed_audio()

    dataset = dataset_input_fn(positives, negatives, backgrounds, NEGATIVE_RATIO, BATCH_SIZE)

    model = nn(input_shape=(SPECTROGRAM_X, 128), n_classes=len(WORDS)+1)

    opt = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)

    model.compile(loss=_soft_f1_macro, optimizer=opt, metrics=["accuracy", f1_scores_1, f1_scores_2, f1_scores_3])

    model.save('{}/model.h5'.format(LOG_DIR))

    csv_logger = tf.keras.callbacks.CSVLogger(str(TRAIN_LOG_FILE))

    cp_callback = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_FILES,
                                                     verbose=1,
                                                     save_weights_only=True,
                                                     period=5)

    model.fit(dataset.make_one_shot_iterator(),
              steps_per_epoch=STEPS_PER_EPOCH,
              validation_data=dataset.make_one_shot_iterator(),
              validation_steps=VALIDATION_STEPS,
              epochs=EPOCHS,
              callbacks=[csv_logger, cp_callback],
              verbose=1
              )
