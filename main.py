from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation

if __name__ == "__main__":
    # Step 1: Ingest
    ingestion = DataIngestion()
    train_data, test_data = ingestion.initiate_data_ingestion()

    # Step 2: Validate
    validator = DataValidation(train_data, test_data)
    validator.validate()
