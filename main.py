from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation

if __name__ == "__main__":
    # Step 1: Ingest
    ingestion = DataIngestion()
    train_data, test_data = ingestion.initiate_data_ingestion()
    print("Data Ingestion Done")

    # Step 2: Validate
    validator = DataValidation(train_data, test_data)
    validator.validate()
    print("Data validation Done")

    # ✅ Step 3: Transform
    transformer = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(train_data, test_data)

    print("✅ Data Transformation Done.")
    print("📁 Preprocessor Saved at:", preprocessor_path)
