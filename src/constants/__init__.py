import os,sys
from datetime import datetime


# artifact -> pipeline folder -> timestamp -> output

def get_current_timestamp():
    return f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"


CURRENT_TIMESTAMP = get_current_timestamp()

ROOT_DIR_KEY = os.getcwd()
DATA_DIR = "Data"
DATA_DIR_KEY = "finalTrain.csv"

ARTIFACT_DIR_KEY = "Artifact"


# Data Ingestion related variables

DATA_INGESTION_KEY = "data_ingestion"
DATA_INGESTION_RAW_DATA_DIR = "raw_data_dir"
DATA_INGESTION_INGESTED_DATA_DIR_KEY = "ingested_dir"
RAW_DATA_DIR_KEY = "raw.csv"
TRAIN_DATA_DIR_KEY = "train.csv"
TEST_DATA_DIR_KEY = "test.csv"
