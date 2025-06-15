import os
import sys
import pandas as pd
from src.logger import logging
from src.excepation import CustomeException

class DataValidation:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def validate(self):
        try:
            logging.info("✅ Starting Data Validation...")

            # Load data
            train_df = pd.read_csv(self.train_data)
            test_df = pd.read_csv(self.test_data)

            # 1. Null values
            logging.info(f"Train Nulls:\n{train_df.isnull().sum()}")
            logging.info(f"Test Nulls:\n{test_df.isnull().sum()}")

            # 2. Duplicates
            logging.info(f"Train Duplicates: {train_df.duplicated().sum()}")
            logging.info(f"Test Duplicates: {test_df.duplicated().sum()}")

            # 3. Data types
            logging.info(f"Train Types:\n{train_df.dtypes}")
            logging.info(f"Test Types:\n{test_df.dtypes}")

            # 4. Summary
            logging.info(f"Train Summary:\n{train_df.describe()}")
            logging.info(f"Test Summary:\n{test_df.describe()}")

            logging.info("✅ Data Validation Completed.")

        except Exception as e:
            raise CustomeException(e, sys)
