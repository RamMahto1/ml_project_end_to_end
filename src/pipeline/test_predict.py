from src.components.predict_pipeline import PredictPipeline

import pandas as pd

def test_prediction():
    data = {
        "feature1": [5.1],
        "feature2": [3.2],
        "feature3": [1.8]
    }
    input_df = pd.DataFrame(data)
    predict_pipeline = PredictPipeline()
    predictions = predict_pipeline.predict(input_df)
    print("Predictions:", predictions)

if __name__ == "__main__":
    test_prediction()
