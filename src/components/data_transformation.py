import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from src.utils import save_object
from src.logger import logging
from src.excepation import CustomeException


@dataclass
class DataTransformationConfig:
    preprocessor_file_path_obj: str = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformation_object(self):
        """
        Creates and returns the data preprocessing pipeline object
        """
        try:
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = [
                'gender', 'race_ethnicity', 'parental_level_of_education',
                'lunch', 'test_preparation_course'
            ]

            numerical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer([
                ("num_pipeline", numerical_pipeline, numerical_columns),
                ("cat_pipeline", categorical_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomeException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads train and test CSVs, applies transformations and saves preprocessor object
        Returns transformed train and test numpy arrays and path to preprocessor
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read the data as data frames")

            preprocessor_obj = self.get_transformation_object()

            target_column_name = "math_score"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessor on training and testing data")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessor object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path_obj,
                obj=preprocessor_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_file_path_obj

        except Exception as e:
            raise CustomeException(e, sys)
