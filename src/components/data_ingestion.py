from src.constants import *
from src.config.configuration import *
import os,sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.config.configuration import *

@dataclass
class DataIngestionConfig:
    train_data_path:str = TRAIN_FILE_PATH
    test_data_path:str = TEST_FILE_PATH
    raw_data_path:str = RAW_FILE_PATH


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initate_data_ingestion(self):
        try:
            logging.info("Loading Data..")
            df = pd.read_csv(DATASET_PATH)
            # df = pd.read_csv(os.path.join())

            logging.info("Create directory for raw data & saving raw data")
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)

        
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False)


            logging.info("Initiating Train-test split")
            train,test = train_test_split(df, test_size=0.25, random_state=101)

            logging.info("Creating directory and saving train & test set")
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            train.to_csv(self.data_ingestion_config.train_data_path, header = True, index=False)

            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path), exist_ok=True)
            test.to_csv(self.data_ingestion_config.test_data_path, header = True, index=False)

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
            

        except Exception as e:
            raise CustomException(e,sys)
        


if __name__ == "__main__":
    logging.info("*** Data ingestion started ***")
    obj = DataIngestion()
    train_path, test_path = obj.initate_data_ingestion()
    
    logging.info("*** Data ingestion completed & Data transformation started ***")

    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.initiate_data_transformation(train_path, test_path)
    
    logging.info("*** Data transformation completed & Model training started***")

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr, test_arr)
    
    logging.info("*** Model training completed ***")