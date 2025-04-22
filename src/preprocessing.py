import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

from src.sliding_windows import SlidingWindowDataset
from torch.utils.data import DataLoader

def data_preprocessing():
    # Load train, validation, and test datasets
    print("Loading processed data...")
    train_data = pd.read_csv(r'data\processed\train_data.csv')
    val_data = pd.read_csv(r'data\processed\val_data.csv')
    test_data = pd.read_csv(r'data\processed\test_data.csv')

    # Reorder columns for consistency
    cols_order = ['id_Serie', 
                  'Accelerometer 1 (m/s^2)', 'Accelerometer 2 (m/s^2)', 'Accelerometer 3 (m/s^2)', 
                  'Microphone (V)', 'Temperature (Celsius)', 'Frequency', 'State', 'Condition']
    train_data = train_data[cols_order]
    val_data = val_data[cols_order]
    test_data = test_data[cols_order]

    # Separate features and target
    X_train = train_data.drop(columns=['Condition'])
    y_train = train_data[['Condition']]
    X_val = val_data.drop(columns=['Condition'])
    y_val = val_data[['Condition']]
    X_test = test_data.drop(columns=['Condition'])
    y_test = test_data[['Condition']]

    # Identify numeric and categorical columns
    num_cols = X_train.select_dtypes(include=np.number).drop(columns='id_Serie').columns.to_list()
    cat_cols = X_train.select_dtypes(exclude=np.number).columns.to_list()

    # Define transformers for numeric and categorical columns
    numeric_transformer = Pipeline(
        steps=[('scaler', RobustScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore", drop='if_binary'))]
    )

    # Combine transformers into a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder='passthrough'
    )

    # Apply preprocessing
    print("Preprocessing data...")
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    # Define column names after transformation
    col_names = ['Accelerometer 1 (m/s^2)', 'Accelerometer 2 (m/s^2)', 'Accelerometer 3 (m/s^2)', 'Microphone (V)', 
                                         'Temperature (Celsius)', 'Frequency', 'State', 'id_Serie']
    
    # Convert to DataFrames and recover original column order (id_Serie last in 'remainder')
    X_train = pd.DataFrame(X_train, columns=col_names)
    X_val = pd.DataFrame(X_val, columns=col_names)
    X_test = pd.DataFrame(X_test, columns=col_names)

    # Reorder columns to bring id_Serie to the front
    def reorder_cols(df):
        cols = df.columns.tolist()
        id_col = [col for col in cols if 'id_Serie' in col]
        reordered = id_col + [col for col in cols if col not in id_col]
        return df[reordered]

    X_train = reorder_cols(X_train)
    X_val = reorder_cols(X_val)
    X_test = reorder_cols(X_test)

    # Encode target variable (Condition) with OneHotEncoder
    enc = OneHotEncoder(sparse_output=False)
    y_train = enc.fit_transform(y_train[['Condition']])
    y_val = enc.transform(y_val[['Condition']])
    y_test = enc.transform(y_test[['Condition']])

    y_train = pd.DataFrame(y_train, columns=enc.get_feature_names_out())
    y_val = pd.DataFrame(y_val, columns=enc.get_feature_names_out())
    y_test = pd.DataFrame(y_test, columns=enc.get_feature_names_out())

    print("Preprocessing was succesful!")
    print("----------------------------------------------------------------")
    # Return all datasets
    return X_train, X_val, X_test, y_train, y_val, y_test, enc