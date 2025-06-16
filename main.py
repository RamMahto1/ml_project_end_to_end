from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Step 1: Ingest Data
ingestion = DataIngestion()
train_path, test_path = ingestion.initiate_data_ingestion()
print("Data Ingestion Done")

# Step 2: Validate Data
validation = DataValidation(train_data=train_path, test_data=test_path)
validation.initiate_data_validation()
print("Data Validation Done")

# Step 3: Transform Data
transformer = DataTransformation()
train_arr, test_arr, _ = transformer.initiate_data_transformation(train_path, test_path)
print("✅ Data Transformation Done.")

# Step 4: Train Model
model_trainer = ModelTrainer()
best_model_name, best_score, best_model = model_trainer.initiate_model_trainer(train_arr, test_arr)
print(f"✅ Best Model: {best_model_name} with R2 score: {best_score}")
