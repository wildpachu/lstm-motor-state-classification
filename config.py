# Data paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
TRAIN_FILE = "train_data.csv"
VAL_FILE = "val_data.csv"
TEST_FILE = "test_data.csv"

# Preprocessing
WINDOW_SIZE = 4200
STRIDE = 2100
BATCH_SIZE = 32

# Training
EPOCHS = 200
LEARNING_RATE = 0.001
PATIENCE = 50
BEST_MODEL_PATH = "models/best_model.pth"

# Model
INPUT_DIM = 7
OUTPUT_DIM = 8
HIDDEN_DIM = 64       
NUM_LAYERS = 2      

# Report
REPORT_PATH = "reports/best_model_report.pdf"
