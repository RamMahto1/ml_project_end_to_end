import os
import sys
from src.logger import logging
from src.excepation import CustomeException  # typo fixed: "excepation" → "exception", "Custome" → "Custom"
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method.")
        try:
            df = pd.read_csv("dataset/exams.csv")  # Use forward slash or raw string
            logging.info("Successfully read the dataset as a DataFrame.")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved to {self.ingestion_config.raw_data_path}")

            train_set, test_set = train_test_split(df, test_size=0.20, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Train-test split complete. Data ingestion finished.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

# Move this to main.py — don't include __main__ block inside class file
