from dotenv import find_dotenv, load_dotenv
from pathlib import Path

load_dotenv(find_dotenv())

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = Path("{}/data".format(PROJECT_DIR))
RAW_DATA_DIR = Path("{}/raw".format(DATA_DIR))
INTERIM_DATA_DIR = Path("{}/interim".format(DATA_DIR))
PROCESSED_DATA_DIR = Path("{}/processed".format(DATA_DIR))

KERNEL_SIZE = 15
STRIDE = 4
FRAME_RATE = 48000
NFFT = 512
TX = FRAME_RATE * 0.0195
FX = int(NFFT / 2) + 1
TY = round((TX - KERNEL_SIZE + STRIDE) / STRIDE)
