import glob
import multiprocessing

from dotenv import find_dotenv, load_dotenv
from pathlib import Path

load_dotenv(find_dotenv())

N_CORES = multiprocessing.cpu_count()

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = Path("{}/data".format(PROJECT_DIR))

RAW_DATA_DIR = Path("{}/raw".format(DATA_DIR))
INTERIM_DATA_DIR = Path("{}/interim".format(DATA_DIR))
PROCESSED_DATA_DIR = Path("{}/processed".format(DATA_DIR))

LOG_DIR = Path("{}/logs".format(PROJECT_DIR))
TRAIN_LOG_FILE = Path("{}/training.log".format(LOG_DIR))
CHECKPOINT_DIR = Path("{}/checkpoints".format(LOG_DIR))
CHECKPOINT_FILES = "{}/cp-{}.ckpt".format(CHECKPOINT_DIR, "{epoch:04d}")
FIGURE_DIR = Path("{}/reports/figures".format(PROJECT_DIR))

WORDS = sorted({Path(k).parent.name for k in glob.glob("{}/positives/*/*.wav".format(RAW_DATA_DIR))})

FRAME_RATE = 16000
WORD_AMPLITUDE = -15

BACKGROUND_AMPLITUDE = -15
BACKGROUND_DURATION_MS = 10 * 1000
NEGATIVE_RATIO = .3
LABEL_DURATION = BACKGROUND_DURATION_MS // 10

FFT_FRAME_LENGTH, FFT_FRAME_STEP, FFT_LENGTH = 1024, 512, 1024
SPECTROGRAM_X = round((BACKGROUND_DURATION_MS / 1000 * FRAME_RATE - FFT_FRAME_LENGTH + FFT_FRAME_STEP) // FFT_FRAME_STEP)
LOWER_EDGE_HERTZ, UPPER_EDGE_HERTZ, NUM_MEL_BINS = 80, 8000, 128

PREFETCH_SIZE = 10
EPOCHS = 10
STEPS_PER_EPOCH = 5
VALIDATION_STEPS = 100
