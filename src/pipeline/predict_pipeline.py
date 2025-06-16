import os
import sys
import pandas as pd
from src.utils import load_object
from src.logger import logging
from src.excepation import CustomeException

class PredictPipeline:
    def __init__(self):
        try:
            # Define paths for the saved preprocessor and model
            self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            self.model_path = os.path.join("artifacts", "model.pkl")
            
            # Load preprocessor and model once when object is created
            logging.info("Loading preprocessor and model for prediction")
            self.preprocessor = load_object(self.preprocessor_path)
            self.model = load_object(self.model_path)
        except Exception as e:
            raise CustomeException(e, sys)
    
    def predict(self, input_df: pd.DataFrame):
        """
        input_df: pandas DataFrame containing the features with same column names as training data.
        Returns: numpy array of predictions.
        """
        try:
            logging.info("Starting prediction process")
            
            # Transform input data using preprocessor (e.g., scaling, encoding)
            input_scaled = self.preprocessor.transform(input_df)
            
            # Predict using the loaded model
            preds = self.model.predict(input_scaled)
            
            logging.info("Prediction successful")
            return preds
        
        except Exception as e:
            raise CustomeException(e, sys)
