import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from tqdm.auto import tqdm
# Main function to process raw files and generate final datasets
def transform_raw_data(db_path: str, output_dir: str):

    # Dictionary mapping condition codes to corresponding frequencies
    freq_dic = {
        '1': 15,
        '2': 30,
        '3': 45,
        '4': 60,
        '5': np.linspace(15, 45, 420000).round(2),
        '6': np.linspace(30, 60, 420000).round(2),
        '7': np.linspace(45, 15, 420000).round(2),
        '8': np.linspace(60, 30, 420000).round(2)
    }

    # Empty DataFrames to store train, validation, and test data
    train_data = pd.DataFrame()
    val_data = pd.DataFrame()
    test_data = pd.DataFrame()

    # Counter to assign a unique ID to each series
    id_serie_count = 0

    print("Processing raw data...")

    # Check if processed data already exist.
    train_file = os.path.join(output_dir, 'train_data.csv')
    val_file = os.path.join(output_dir, 'val_data.csv')
    test_file = os.path.join(output_dir, 'test_data.csv')

    if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
        print("Processed files already exist. Skipping raw data transformation.")
        print("----------------------------------------------------------------")
        return

    # Loop through each folder in the raw data directory
    for folder in os.listdir(db_path):
        folder_path = os.path.join(db_path, folder)  # Full path to the current folder

        # Loop through each file in the folder
        for archivo in tqdm(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, archivo)  # Full path to the current file

            data_file = pd.read_csv(file_path)  # Read the CSV file into a DataFrame

            # Use regex to extract information from the filename (condition, frequency, state)
            match = re.match(r"^([A-Z])_([A-Z])_(\d)_(\d)\.csv$", archivo)

            # Add new columns: Series ID, Condition, Frequency, and State
            data_file["id_Serie"] = id_serie_count
            data_file["Condition"] = match.group(1) + match.group(2)
            data_file["Frequency"] = freq_dic[match.group(3)]
            data_file["State"] = "On Load" if int(match.group(4)) == 1 else "Idle"

            # Split into Train (70%) and Test + Validation (30%)
            train, test_val = train_test_split(data_file, shuffle=False, test_size=0.3)

            # Split Test + Validation into Validation (10%) and Test (20%)
            val, test = train_test_split(test_val, shuffle=False, test_size=2/3)  # 2/3 of 30% is 20%

            # Append the split data to the corresponding global DataFrames
            train_data = pd.concat([train_data, train])
            val_data = pd.concat([val_data, val])
            test_data = pd.concat([test_data, test])

            # Increment series ID counter
            id_serie_count += 1

    # Reset indices in final DataFrames
    train_data.reset_index(inplace=True, drop=True)
    val_data.reset_index(inplace=True, drop=True)
    test_data.reset_index(inplace=True, drop=True)

    # Create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save final DataFrames as CSV files
    print("Saving processed data...")
    train_data.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
    val_data.to_csv(os.path.join(output_dir, 'val_data.csv'), index=False)
    test_data.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)

    print("Data processing and saving completed successfully.")
    print("----------------------------------------------------------------")