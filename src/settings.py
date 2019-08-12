import glob
import multiprocessing
import os

from dotenv import find_dotenv, load_dotenv
from pathlib import Path

load_dotenv(find_dotenv())

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = Path("{}/data".format(PROJECT_DIR))
RAW_DATA_DIR = Path("{}/raw".format(DATA_DIR))
INTERIM_DATA_DIR = Path("{}/interim".format(DATA_DIR))

PROCESSED_DATA_DIR = Path("{}/processed".format(DATA_DIR))

TRIGGER_PROCESSED_DATA_DIR = Path("{}/trigger".format(PROCESSED_DATA_DIR))
DEV_TRIGGER_PROCESSED_DATA_DIR = Path("{}/dev".format(TRIGGER_PROCESSED_DATA_DIR))
VAL_TRIGGER_PROCESSED_DATA_DIR = Path("{}/val".format(TRIGGER_PROCESSED_DATA_DIR))


CLASSIF_PROCESSED_DATA_DIR = Path("{}/classif".format(PROCESSED_DATA_DIR))
DEV_CLASSIF_PROCESSED_DATA_DIR = Path("{}/dev".format(CLASSIF_PROCESSED_DATA_DIR))
VAL_CLASSIF_PROCESSED_DATA_DIR = Path("{}/val".format(CLASSIF_PROCESSED_DATA_DIR))

LOG_DIR = Path("{}/logs".format(PROJECT_DIR))

TRIGGER_LOG_DIR = Path("{}/trigger".format(LOG_DIR))
TRIGGER_TRAINING_LOG_FILE = Path("{}/training.log".format(TRIGGER_LOG_DIR))
TRIGGER_CHECKPOINT_DIR = Path("{}/checkpoints".format(TRIGGER_LOG_DIR))
TRIGGER_CHECKPOINT_FILES = "{}/cp-{}.ckpt".format(TRIGGER_CHECKPOINT_DIR, "{epoch:04d}")

CLASSIF_LOG_DIR = Path("{}/classif".format(LOG_DIR))
CLASSIF_TRAINING_LOG_FILE = Path("{}/training.log".format(CLASSIF_LOG_DIR))
CLASSIF_CHECKPOINT_DIR = Path("{}/checkpoints".format(CLASSIF_LOG_DIR))
CLASSIF_CHECKPOINT_FILES = "{}/cp-{}.ckpt".format(CLASSIF_CHECKPOINT_DIR, "{epoch:04d}")

N_CORES = multiprocessing.cpu_count()

N_SAMPLES_IN_TFRECORD = 100
SAMPLE_DURATION_MS = 5000
LABEL_DURATION = 25
CROSSFADE_MS = 100
N_TRIGGER_DEV_SAMPLES = 400
N_TRIGGER_VAL_SAMPLES = int(N_TRIGGER_DEV_SAMPLES / 4)

KERNEL_SIZE = 15
STRIDE = 4
FRAME_RATE = 44100
NFFT = 512
FX = int(NFFT / 2) + 1
TX = int(FRAME_RATE * (SAMPLE_DURATION_MS / 1000) / NFFT) * 2
TY = round((TX - KERNEL_SIZE) / STRIDE) + 1

MULTRIGGER_MODE = False

N_WORDS = len({Path(k).parent for k in glob.glob("{}/positives/*/*.wav".format(RAW_DATA_DIR))}) + 1

if MULTRIGGER_MODE:
    N_CLASSES = N_WORDS
else:
    N_CLASSES = 2

TRIGGER_EPOCHS = 20
TRIGGER_BATCH_SIZE = 64

N_CLASSIF_DEV_SAMPLES = 1000
N_CLASSIF_VAL_SAMPLES = int(N_CLASSIF_DEV_SAMPLES / 4)

CHUNK_DURATION = 0.5  # Each read length in seconds from mic.
FS = 48000  # sampling rate for mic
CHUNK_SAMPLES = int(FS * CHUNK_DURATION)  # Each read length in number of samples.

# Each model input data duration in seconds, need to be an integer numbers of chunk_duration
FEED_DURATION = 10
FEED_SAMPLES = int(FS * FEED_DURATION)

assert FEED_DURATION / CHUNK_DURATION == int(FEED_DURATION / CHUNK_DURATION)
