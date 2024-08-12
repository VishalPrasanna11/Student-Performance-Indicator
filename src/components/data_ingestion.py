#Data Ingestion will play very important role in the project.
# It will be responsible for reading the data from different sources and converting it into a format that can be used for further processing.
# The data ingestion module will have the following functionalities:
# 1. Read data from different sources like CSV, Excel, Database, etc.
# 2. Perform data validation and cleaning.
# 3. Save the data in a format that can be used for further processing.
# 4. Log the data ingestion process.
# 5. Handle exceptions that occur during data ingestion.
# 6. Provide a summary of the data ingestion process.


import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join( "artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        try:
            logging.info("Data Ingestion process started")
            df = pd.read_csv('notebook/data/StudentsPerformance.csv')
            logging.info("Data Ingestion process completed successfully")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)
            logging.info("Raw data saved successfully")
            logging.info("Train and Test data split started")
            train_set,test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)
            logging.info("Train and Test data split completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                # self.ingestion_config.raw_data_path
            )
        except Exception as e:
            logging.error("Data Ingestion process failed")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()