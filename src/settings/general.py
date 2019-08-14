import multiprocessing
import glob

from dotenv import find_dotenv, load_dotenv
from pathlib import Path

load_dotenv(find_dotenv())

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = Path("{}/data".format(PROJECT_DIR))
RAW_DATA_DIR = Path("{}/raw".format(DATA_DIR))
INTERIM_DATA_DIR = Path("{}/interim".format(DATA_DIR))

PROCESSED_DATA_DIR = Path("{}/processed".format(DATA_DIR))

LOG_DIR = Path("{}/logs".format(PROJECT_DIR))
FIGURE_DIR = Path("{}/reports/figures".format(PROJECT_DIR))

N_CORES = multiprocessing.cpu_count()

N_WORDS = len({Path(k).parent for k in glob.glob("{}/positives/*/*.wav".format(RAW_DATA_DIR))})

FRAME_RATE = 44100
NFFT = 512