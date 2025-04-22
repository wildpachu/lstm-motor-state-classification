# ⚙️ LSTM-Based Motor State Classification ⚙️
This project utilizes a Long Short-Term Memory (LSTM) network to classify motor states based on sensor data. The goal is to classify the current state of a motor by analyzing time-series sensor data. The model is trained using a sliding window technique and evaluated with various performance metrics.

The project workflow includes the following steps:

1. **Data Preprocessing 🧹**: Transforming raw sensor data into a format suitable for training.
2. **Sliding Window Technique**: Creating sequences from time-series data to use as inputs for the LSTM.
3. **Model Training 🤖**: Training the LSTM network to classify motor states.
4. **Model Evaluation 📊**: Evaluating the model with test data to assess its performance.
5. **Report Generation 📄**: Generating a PDF report with evaluation metrics, confusion matrix, and training performance plots.
  
## Table of Contents 📚

- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#model-training)
- [Results](#results)
- [License](#license)

## Installation 🛠️

### Prerequisites 🔑

Make sure you have the following tools installed:

- Python 3.7+ (recommended) 🐍
- pip to install dependencies 📦

### Steps 🔽

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/lstm-motor-state-classification.git
    cd lstm-motor-state-classification
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage 🚀

### Step 1: Data Download⬇

To download raw data, run the following command:

```bash
python download_data.py
```
### Step 2: Run the Main Script📈

To train the LSTM model, execute the following command:

```bash
python main.py
```
This will:

1. Preprocess the data in order to fit the training
2. Train the model using the preprocessed data.
3. Show the loss and accuracy during training.
4. Save the best model based on performance.
5. Generate a PDF report with evaluation metrics.

### Step 3: Check the Results 🏆
After training, a PDF report will be generated and saved in the `reports` directory. 

## Model Training 🏋️‍♂️
Below are the key configuration parameters that define how the model is trained, preprocessed, and evaluated.

### Configuration Parameters 🔧

The configuration settings are stored in the `config.py` file, where you can modify various parameters related to data paths, preprocessing, model training, and the generation of the report. Here's an overview of the parameters:

#### Data Paths 📂
- **RAW_DATA_DIR**: Directory where raw data is stored (default: `data/raw`).
- **PROCESSED_DATA_DIR**: Directory where preprocessed data is saved (default: `data/processed`).
- **TRAIN_FILE**: Filename for training data (default: `train_data.csv`).
- **VAL_FILE**: Filename for validation data (default: `val_data.csv`).
- **TEST_FILE**: Filename for test data (default: `test_data.csv`).

#### Preprocessing 🔄
- **WINDOW_SIZE**: The size of the sliding window used to split time-series data (default: `4200`).
- **STRIDE**: The stride (step size) between consecutive windows (default: `2100`).
- **BATCH_SIZE**: The batch size used during model training (default: `32`).

#### Training ⚡
- **EPOCHS**: The number of epochs for model training (default: `200`).
- **LEARNING_RATE**: The learning rate used by the optimizer (default: `0.001`).
- **PATIENCE**: The number of epochs to wait for improvement before stopping early (default: `50`).
- **BEST_MODEL_PATH**: Path to save the best-performing model during training (default: `models/best_model.pth`).

#### Model 🧠
- **INPUT_DIM**: The number of input features for the LSTM model (default: `7`).
- **OUTPUT_DIM**: The number of output classes to predict (default: `8`).
- **HIDDEN_DIM**: The number of hidden units in each LSTM layer (default: `64`).
- **NUM_LAYERS**: The number of layers in the LSTM model (default: `2`).

#### Report 📄
- **REPORT_PATH**: The path where the PDF report will be saved (default: `reports/best_model_report.pdf`).

## Results 📑
Once the model is trained and evaluated, the results are included in the PDF report. The report includes:

* **Learning Curves📈**: Visual representation of loss and accuracy during training.

* **Model Accuracy 📊**: Overall accuracy of the trained model.

* **Confusion Matrix 🔴**: Visual representation of prediction results.

* **Classification Report 📝**: Detailed precision, recall, and F1-score for each motor state.

## License 📄

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
