import glob
import multiprocessing

from dotenv import find_dotenv, load_dotenv
from pathlib import Path

load_dotenv(find_dotenv())

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = Path("{}/data".format(PROJECT_DIR))
RAW_DATA_DIR = Path("{}/raw".format(DATA_DIR))
INTERIM_DATA_DIR = Path("{}/interim".format(DATA_DIR))
PROCESSED_DATA_DIR = Path("{}/processed".format(DATA_DIR))
DEV_PROCESSED_DATA_DIR = Path("{}/dev".format(PROCESSED_DATA_DIR))
VAL_PROCESSED_DATA_DIR = Path("{}/val".format(PROCESSED_DATA_DIR))

LOG_DIR = Path("{}/logs".format(PROJECT_DIR))
TRAIN_LOG_FILE = Path("{}/training.log".format(LOG_DIR))
CHECKPOINT_DIR = Path("{}/checkpoints".format(LOG_DIR))
CHECKPOINT_FILES = "{}/cp-{}.ckpt".format(CHECKPOINT_DIR, "{epoch:04d}")

N_CORES = multiprocessing.cpu_count()

SAMPLE_DURATION_MS = 5000
LABEL_DURATION = 25
CROSSFADE_MS = 100
N_DEV_SAMPLES = 400
N_VAL_SAMPLES = int(N_DEV_SAMPLES / 4)

KERNEL_SIZE = 15
STRIDE = 4
FRAME_RATE = 48000
NFFT = 512
TX = int(FRAME_RATE * 0.0195)
FX = int(NFFT / 2) + 1
TY = round((TX - KERNEL_SIZE + STRIDE) / STRIDE)

MULTRIGGER_MODE = False

N_WORDS = len({Path(k).parent for k in glob.glob("{}/positives/*/*.wav".format(RAW_DATA_DIR))}) + 1

if MULTRIGGER_MODE:
    N_CLASSES = N_WORDS
else:
    N_CLASSES = 2

EPOCHS = 20
BATCH_SIZE = 64

CHUNK_DURATION = 0.5  # Each read length in seconds from mic.
FS = 48000  # sampling rate for mic
CHUNK_SAMPLES = int(FS * CHUNK_DURATION)  # Each read length in number of samples.

# Each model input data duration in seconds, need to be an integer numbers of chunk_duration
FEED_DURATION = 10
FEED_SAMPLES = int(FS * FEED_DURATION)

assert FEED_DURATION / CHUNK_DURATION == int(FEED_DURATION / CHUNK_DURATION)
