from src.pipeline.predict_pipeline import CustomData, PredictPipeline

def run_prediction_test():
    try:
        # Step 1: Create sample input
        input_data = CustomData(
            gender="female",
            race_ethnicity="group C",
            parental_level_of_education="some college",
            lunch="standard",
            test_preparation_course="completed",
            reading_score=70,
            writing_score=90
        )

        # Step 2: Convert to DataFrame
        df = input_data.get_data_as_data_frame()
        print("📦 Input DataFrame:")
        print(df)

        # Step 3: Predict
        pipeline = PredictPipeline()
        prediction = pipeline.predict(df)

        print("\n🔮 Prediction Result:", prediction)

    except Exception as e:
        print("❌ Error during prediction:", str(e))


if __name__ == "__main__":
    run_prediction_test()
