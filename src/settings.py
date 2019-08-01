import glob

from dotenv import find_dotenv, load_dotenv
from pathlib import Path

load_dotenv(find_dotenv())

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = Path("{}/data".format(PROJECT_DIR))
RAW_DATA_DIR = Path("{}/raw".format(DATA_DIR))
INTERIM_DATA_DIR = Path("{}/interim".format(DATA_DIR))
PROCESSED_DATA_DIR = Path("{}/processed".format(DATA_DIR))

LOG_DIR = Path("{}/logs".format(PROJECT_DIR))
TRAIN_LOG_FILE = Path("{}/training.log".format(LOG_DIR))
CHECKPOINT_DIR = Path("{}/checkpoints".format(LOG_DIR))
CHECKPOINT_FILES = "{}/cp-{}.ckpt".format(CHECKPOINT_DIR, "{epoch:04d}")

SAMPLE_DURATION_MS = 5000
N_SAMPLES = 300

KERNEL_SIZE = 15
STRIDE = 4
FRAME_RATE = 48000
NFFT = 512
TX = int(FRAME_RATE * 0.0195)
FX = int(NFFT / 2) + 1
TY = round((TX - KERNEL_SIZE + STRIDE) / STRIDE)

MULTRIGGER_MODE = True

if MULTRIGGER_MODE:
    N_CLASSES = len({Path(k).parent for k in glob.glob("{}/positives/*/*.wav".format(RAW_DATA_DIR))}) + 1
else:
    N_CLASSES = 1
