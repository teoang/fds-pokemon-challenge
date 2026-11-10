# ==============================================================================
# STEP 1A: IMPORTS, CONFIGURATION & FILE PATHS
# ==============================================================================
import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# If using Google Drive on Colab, uncomment the next two lines:
# from google.colab import drive
# drive.mount('/content/drive')

# Set your base directory path (keep the same path structure you use on Colab/Kaggle)
DATA_DIR = "/content/drive/MyDrive/1 SEMESTRE DATA SCIENCE/FUNDAMENTALS OF DATA SCIENCE/FDS PROJECT"  # Change to your Drive/Kaggle folder if needed, e.g., "/content/drive/MyDrive/FDS_Pokemon/"

TYPE_CHART_PATH = os.path.join(DATA_DIR, "type_chart_gen2.csv")
EXCEL_UNPACKED_PATH = os.path.join(DATA_DIR, "pokemon_json_spacchettato.xlsx")
TRAIN_JSONL_PATH = os.path.join(DATA_DIR, "train.jsonl")
TEST_JSONL_PATH = os.path.join(DATA_DIR, "test.jsonl")

ALL_TYPES = [
    'normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting',
    'poison', 'ground', 'flying', 'psychic', 'bug', 'rock', 'ghost',
    'dragon', 'dark', 'steel'
]