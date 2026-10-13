
# Initialization of packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import sklearn
import seaborn as sns
import statsmodels.api as sm
import glob
import os
import re


from xgboost import XGBClassifier

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, mean_squared_error,f1_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from ModelingFunctions.GEMpythonlib.GEMpythonfunctions import *
from statsmodels.tsa.stattools import acf

from hmmlearn.hmm import GaussianHMM

def load_data(file_path):
    # Opening files
    file_list = os.listdir(file_path)

    # Array or dataframe initialization
    data = []
    # Initialize an empty list to collect DataFrames
    data_frames = []

    for file in file_list:
        # Read each file into a DataFrame and append to the list
        monthly_data = pd.read_csv(file_path +"/"+ file)
        data_frames.append(monthly_data)

    # Concatenate all DataFrames in the list at once
    data = pd.concat(data_frames, ignore_index=True)

    # Display the final concatenated DataFrame
    # display(data)

    # Sort values by 'Time' in ascending order
    data.sort_values('Time', ascending=True, inplace=True)

    # Now you can proceed to drop the duplicates as before
    data.drop_duplicates(subset='Time', inplace=True)

    # Reset the index after dropping duplicates
    data.reset_index(drop=True, inplace=True)

    # Display the cleaned data
    # display(data)

    #should have the same datetime format
    data['Time'] = pd.to_datetime(data['Time'], format="%m/%d/%y %H:%M")

    # Check interval of data:

    is_valid, invalid_data = check_time_intervals(data)

    # If there are invalid rows, display them
    if not is_valid:
        print("Invalid rows:")
        # display(invalid_data)

    return data

def train_test_split(data):
    # Determine the split index (60% for training)
    split_index = int(len(data) * 0.6)

    # Split the data into training and testing sets
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    # Display the sizes of the train and test sets
    print(f'Training data size: {len(train_data)}')
    print(f'Test data size: {len(test_data)}')

    # Optionally, display the first few rows of each
    # print("Training Data Sample:")
    # display(train_data)

    # print("Test Data Sample:")
    # display(test_data)

    return train_data, test_data

def evaluate_model_performance_model(y_true, y_pred):
    """
    Function to evaluate model performance using accuracy, confusion matrix, precision, and recall.
    """
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1_metric = f1_score(y_true,y_pred,zero_division = 0)


    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

    performance_metrics = {
        'Accuracy': accuracy,
        'Confusion Matrix': conf_matrix,
        'Precision': precision,
        'Recall': recall,
        'F1_Score':f1_metric
    }


    return performance_metrics



def generate_group3features(train_data, test_data, columns_to_compare, freq_a, freq_b, volt_a, volt_b):
    high_freq_threshold = 60
    low_freq_threshold = 40

    high_volt_threshold = 250
    low_volt_threshold = 2

    voltage_delta = 10

    train_data_base = train_data.copy()
    train_data_ts = train_data.copy()
    test_data_ts = test_data.copy()

    ## FEATURE GENERATION
    # Using mean to compare
    train_data_ts, test_data_ts = calculate_mean_flags(train_data_ts, test_data_ts, train_data_base, columns_to_compare)

    # Assuming train_data_ts and test_data_ts are your DataFrames
    train_data_ts, test_data_ts = process_time_series_data(train_data_ts, test_data_ts, 
                                                            freq_lab_column= freq_a,
                                                            freq_clinique_column= freq_b,
                                                            volt_lab_column= volt_a,
                                                            volt_clinique_column= volt_b,
                                                            high_freq_threshold = high_freq_threshold,
                                                            low_freq_threshold = low_freq_threshold, 
                                                            high_volt_threshold = high_volt_threshold, 
                                                            low_volt_threshold = low_volt_threshold)
    # display(train_data_ts)
    # display(test_data_ts)
    return train_data_ts, test_data_ts

def display_generator_statistics(train_data_ts_rf):
    # Ensure columns are boolean (if not already)
    actual_minutes = train_data_ts_rf['Generator_ON'].astype(bool).sum()
    predicted_minutes = train_data_ts_rf['prediction'].astype(bool).sum()

    # Total duration in hours (assuming each row represents 1 minute)
    total_hours = len(train_data_ts_rf) / 60

    # Display actual and predicted minutes with equivalent hours
    print(
        f'Actual Minutes: {actual_minutes} minutes, or {actual_minutes / 60:.2f} hours within {total_hours:.2f} hours')
    print(
        f'Predicted Minutes: {predicted_minutes} minutes, or {predicted_minutes / 60:.2f} hours within {total_hours:.2f} hours')
    
    return actual_minutes, predicted_minutes


def generate_group4features(train_data_ts, test_data_ts, freq_a, freq_b, volt_a, volt_b):
    # Insert variables
    n_states = 5
    hmm_columns = [freq_a, freq_b, volt_a, volt_b]

    # Make copies of original train and test sets
    train_data_ts_co = train_data_ts.copy()
    test_data_ts_co = test_data_ts.copy()

    # Sort by Time
    train_data_ts_co.sort_values('Time', ascending=True, inplace=True)
    test_data_ts_co.sort_values('Time', ascending=True, inplace=True)

    # Clean NaNs, infs, and low-variance columns
    valid_hmm_columns = []

    for col in hmm_columns:
        if col in train_data_ts_co.columns and col in test_data_ts_co.columns:
            values_train = train_data_ts_co[col].replace([np.inf, -np.inf], 0).fillna(0)
            values_test = test_data_ts_co[col].replace([np.inf, -np.inf], 0).fillna(0)

            if values_train.std() < 1e-3:
                print(f"⚠️ Column {col} has too low variance (std={values_train.std():.5f}), skipping HMM fitting.")
            else:
                train_data_ts_co[col] = values_train
                test_data_ts_co[col] = values_test
                valid_hmm_columns.append(col)

    hmm_columns = valid_hmm_columns  # update hmm_columns

    # Train HMMs on train set
    train_data_ts_co, hmm_models, hmm_features = generate_hmm(
        train_data_ts_co,
        hmm_columns,
        n_states=n_states
    )

    # Apply HMMs to test set
    def apply_hmm(df, columns, hmm_models):
        """
        Apply trained HMM models to a new dataset and generate one-hot encoded HMM states.

        Parameters:
        - df (DataFrame): The DataFrame to apply HMMs to.
        - columns (list): List of feature columns to apply HMMs.
        - hmm_models (dict): Trained HMM models from the training set.

        Returns:
        - df_result (DataFrame): DataFrame with added HMM states and one-hot encoded features.
        """
        df_result = df.copy()
        hmm_feature_cols = []

        for col in columns:
            if col not in df_result.columns:
                print(f"⚠️ Column {col} missing in DataFrame. Skipping...")
                continue


            X = df_result[col].replace([np.inf, -np.inf], 0).fillna(0).to_numpy().reshape(-1, 1)

            model = hmm_models.get(col)
            state_col = f'HMM_State_{col}'

            if model is not None:
                # X = df_result[col].values
                df_result[state_col] = model.predict(X)

                dummies = pd.get_dummies(df_result[state_col], prefix=state_col)

                for i in range(n_states):
                    expected_col = f'{state_col}_{i}'
                    if expected_col not in dummies.columns:
                        dummies[expected_col] = 0

                dummies = dummies[[f'{state_col}_{i}' for i in range(n_states)]]
                df_result = pd.concat([df_result, dummies], axis=1)
                hmm_feature_cols.extend(dummies.columns.tolist())
            else:
                print(f"⚠️ No trained HMM model found for {col}. Skipping...")

        return df_result

        #     # if col not in df_result.columns:
        #     #     print(f"⚠️ Column {col} missing in DataFrame. Skipping...")
        #     #     continue

        #     X = df_result[col].replace([np.inf, -np.inf], 0).fillna(0).to_numpy().reshape(-1, 1)
        #     state_col = f'HMM_State_{col}'
        #     model = hmm_models.get(col)

        #     if model is not None:
        #         df_result[state_col] = model.predict(X)

        #         # One-hot encode HMM state
        #         dummies = pd.get_dummies(df_result[state_col], prefix=state_col)
        #         df_result = pd.concat([df_result, dummies], axis=1)
        #         hmm_feature_cols.extend(dummies.columns.tolist())
        #     else:
        #         print(f"⚠️ No trained HMM model found for {col}. Skipping...")

        # return df_result
    
    test_data_ts_co = apply_hmm(test_data_ts_co, hmm_columns, hmm_models)

    return train_data_ts_co, test_data_ts_co,hmm_features