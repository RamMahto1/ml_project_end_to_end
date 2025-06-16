import sys
import os
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from src.utils import evaluate_models, save_object
from src.logger import logging
from src.excepation import CustomeException

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "SVR": SVR()
            }

            params = {
                "LinearRegression": {},
                "Ridge": {"alpha": [0.1, 1.0, 10.0]},
                "Lasso": {"alpha": [0.01, 0.1, 1.0]},
                "ElasticNet": {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.1, 0.5, 0.9]},
                "DecisionTree": {"max_depth": [3, 5, 7, None]},
                "RandomForest": {"n_estimators": [50, 100], "max_depth": [5, 10, None]},
                "GradientBoosting": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
                "SVR": {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10]}
            }

            logging.info("Evaluating models")
            best_model_name, best_score, best_model = evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )

            logging.info(f"Best model: {best_model_name} with R2 score: {best_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name, best_score, best_model


        except Exception as e:
            raise CustomeException(e, sys)
