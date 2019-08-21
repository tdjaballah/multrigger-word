import logging
import tensorflow as tf

from src.utils.misc_utils import clean_data_dir
from src.utils.audio import normalize_volume
from src.settings.general import *


def main():

    files_to_delete = glob.glob("{}/**/*.wav".format(PROCESSED_DATA_DIR))
    clean_data_dir(files_to_delete)

    normalize_volume()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    tf.enable_eager_execution()
    main()
