# Initialization of packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import openpyxl
# import sklearn
# import seaborn as sns
# import statsmodels.api as sm
# import glob
# import os


# from xgboost import XGBClassifier

# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, mean_squared_error

# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV

# from sklearn.ensemble import RandomForestClassifier

# Constants
NOx_lb_hp_hr = 0.024
CO_lb_hp_hr = 0.0055
SOx_lb_hp_hr = 0.00809
CO2_lb_hp_hr = 1.16
PM_lb_hp_hr = 0.0007

# Functions
def check_time_intervals(data, time_column='Time', interval_minutes=2):
    """
    Checks if all rows in the DataFrame are separated by a specified time interval.

    Parameters:
    - data: The DataFrame containing the time data.
    - time_column: The column name that contains the time data (default is 'Time').
    - interval_minutes: The time interval in minutes to check for (default is 2 minutes).

    Returns:
    - True if all rows are separated by the specified interval, False otherwise.
    - A DataFrame showing rows that do not follow the interval.
    """

    # Ensure the time column is in datetime format
    data[time_column] = pd.to_datetime(data[time_column])

    # Sort the data by time in case it's not already sorted
    data = data.sort_values(by=time_column)

    # Calculate the time difference between consecutive rows
    data['time_diff'] = data[time_column].diff()

    # Define the expected time difference
    expected_diff = pd.Timedelta(minutes=interval_minutes)

    # Identify rows where the time difference is not equal to the expected difference
    invalid_rows = data[data['time_diff'] != expected_diff]

    if invalid_rows.empty:
        print("All rows are separated by exactly {} minutes.".format(interval_minutes))
        return True, None
    else:
        print("Some rows are not separated by exactly {} minutes.".format(interval_minutes))
        return False, invalid_rows


# Usage
# is_valid, invalid_data = check_time_intervals(data)

# If there are invalid rows, display them
# if not is_valid:
#    print("Invalid rows:")
#    display(invalid_data)


#####################################################################################################


def calculate_delta(df, column_name, new_col_name):
    """
    Calculate the delta (difference) between the current and previous values of a column,
    and store the result in a new column.

    Parameters:
    - df: The DataFrame.
    - column_name: The column for which the delta is calculated.
    - new_col_name: The name of the new column where the delta will be stored.

    Returns:
    - The DataFrame with the new delta column.
    """
    df.loc[:, new_col_name] = df[column_name] - df[column_name].shift(1)
    return df


# Example usage:
# train_data = calculate_delta(train_data, 'Frequency 3E2F4FD3 (Ch Cbca Bethesda Ndosho - laboratoire)', 'Freq_delta_lab')
# train_data = calculate_delta(train_data, 'Frequency C30F2E03 (Ch Cbca Bethesda Ndosho - clinique)', 'Freq_delta_clinique')


#####################################################################################################


def evaluate_model_performance(df, actual_column='Generator_ON', prediction_column='prediction'):
    """
    Function to evaluate model performance using accuracy, confusion matrix, precision, and recall.

    Parameters:
    df (DataFrame): The DataFrame containing the actual and predicted values.
    actual_column (str): The column name for the actual values. Default is 'Generator_ON'.
    prediction_column (str): The column name for the predicted values. Default is 'prediction'.

    Returns:
    dict: A dictionary containing the accuracy, confusion matrix, precision, and recall.
    """

    # Extract the actual and predicted values
    y_true = df[actual_column]
    y_pred = df[prediction_column]

    # Calculate the accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Calculate precision
    precision = precision_score(y_true, y_pred, zero_division=0)

    # Calculate recall
    recall = recall_score(y_true, y_pred, zero_division=0)

    # Create a dictionary to store all the metrics
    performance_metrics = {
        'Accuracy': accuracy,
        'Confusion Matrix': conf_matrix,
        'Precision': precision,
        'Recall': recall
    }

    # Print the metrics
    print(f'Accuracy of model: {performance_metrics["Accuracy"]}')
    print(f'Confusion Matrix:\n{performance_metrics["Confusion Matrix"]}')
    print(f'Precision of model: {performance_metrics["Precision"]}')
    print(f'Recall of model: {performance_metrics["Recall"]}')

    return performance_metrics


# Example usage:
# metrics = evaluate_model_performance(train_data_base)


#############################################################################


def evaluate_model_performance_rf(y_true, y_pred):
    """
    Function to evaluate model performance using accuracy, confusion matrix, precision, and recall.
    """
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')


##############################################################################


def plot_predictions_vs_actuals(data, model_name, time_column='Time', actual_column='Generator_ON',
                                prediction_column='prediction',
                                freq_lab_column='Frequency 3E2F4FD3 (Ch Cbca Bethesda Ndosho - laboratoire)',
                                freq_clinique_column='Frequency C30F2E03 (Ch Cbca Bethesda Ndosho - clinique)',
                                volt_lab_column='Voltage 3E2F4FD3 (Ch Cbca Bethesda Ndosho - laboratoire)',
                                volt_clinique_column='Voltage C30F2E03 (Ch Cbca Bethesda Ndosho - clinique)',
                                include_voltage=True, include_frequency=True):
    """
    Function to plot actual vs predicted Generator_ON with optional frequency and voltage data.

    Parameters:
    - data (DataFrame): The DataFrame containing the data.
    - model_name (str): The name of the model (for the plot title).
    - time_column (str): The name of the time column.
    - actual_column (str): The name of the actual Generator_ON column.
    - prediction_column (str): The name of the predicted Generator_ON column.
    - freq_lab_column (str): The column name for lab frequency.
    - freq_clinique_column (str): The column name for clinic frequency.
    - volt_lab_column (str): The column name for lab voltage.
    - volt_clinique_column (str): The column name for clinic voltage.
    - include_voltage (bool): Whether to include voltage data in the plot.
    - include_frequency (bool): Whether to include frequency data in the plot.
    """

    # Step 1: Ensure the 'Time' column is in datetime format
    data[time_column] = pd.to_datetime(data[time_column], format='%Y-%m-%d %H:%M:%S')

    # Step 2: Sort the data by 'Time' to ensure the time series is ordered correctly
    data = data.sort_values(by=time_column)

    # Create the figure and axis for Generator_ON
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Scatter plot for correct and incorrect predictions
    correct_predictions = data[data[actual_column] == data[prediction_column]]
    incorrect_predictions = data[data[actual_column] != data[prediction_column]]

    # Scatter plot for correctly predicted points
    ax1.scatter(correct_predictions[time_column], correct_predictions[actual_column], label='Correct Prediction',
                color='green', marker='o')

    # Scatter plot for incorrectly predicted points
    ax1.scatter(incorrect_predictions[time_column], incorrect_predictions[actual_column], label='Incorrect Prediction',
                color='red', marker='x')

    # Set labels for the first axis (Generator_ON)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Generator_ON', color='black')
    ax1.set_title(f'Actual vs Predicted Generator_ON with Frequency and Voltage ({model_name})')
    ax1.tick_params(axis='y', labelcolor='black')
    plt.xticks(rotation=45)

    # Plot for Frequency and Voltage using a secondary y-axis if specified
    if include_voltage or include_frequency:
        ax2 = ax1.twinx()  # Create a second y-axis

        # Plot frequency if specified
        if include_frequency:
            ax2.plot(data[time_column], data[freq_lab_column], label='Frequency Lab', color='blue', linestyle='--')
            ax2.plot(data[time_column], data[freq_clinique_column], label='Frequency Clinique', color='purple',
                     linestyle='--')

        # Plot voltage if specified
        if include_voltage:
            ax2.plot(data[time_column], data[volt_lab_column], label='Voltage Lab', color='orange', linestyle='--')
            ax2.plot(data[time_column], data[volt_clinique_column], label='Voltage Clinique', color='brown',
                     linestyle='--')

        # Set the y-axis limits for frequency and voltage if either is plotted
        ax2.set_ylim(0, 270)  # Voltage and frequency common limit (0-250)
        ax2.set_yticks(range(0, 271, 50))  # Set ticks for every 50 units

        # Set the label for the secondary axis (Frequency and Voltage)
        ax2.set_ylabel('Frequency (Hz) / Voltage (V)', color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        # Combine legends from both axes
        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    # Show the plot
    plt.tight_layout()
    plt.show()


# Example usage for different models:

# For Random Forest Model with both voltage and frequency
# lot_predictions_vs_actuals(train_data_rf, model_name="Random Forest (Train Set)", include_voltage=True, include_frequency=True)

# For Baseline Model without voltage
# plot_predictions_vs_actuals(baseline_data, model_name="Baseline Model", include_voltage=False, include_frequency=True)

# For Mean Model without frequency
# plot_predictions_vs_actuals(mean_model_data, model_name="Mean Model", include_voltage=True, include_frequency=False)


##############################################################################


def one_hot_encode_hour(df, time_column):
    """
    Function to one-hot encode the hour of the day from a time column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the time column.
    time_column (str): The name of the time column (in datetime format).

    Returns:
    pd.DataFrame: The DataFrame with one-hot encoded hours added.
    """
    # Ensure the time column is in datetime format
    df[time_column] = pd.to_datetime(df[time_column])

    # Extract the hour from the time column
    df['Hour'] = df[time_column].dt.hour

    # Perform one-hot encoding for the 'Hour' column
    hour_dummies = pd.get_dummies(df['Hour'], prefix='Hour')

    # Concatenate the original dataframe with the one-hot encoded hour data
    df = pd.concat([df, hour_dummies], axis=1)

    # Drop the 'Hour' column if not needed
    df.drop('Hour', axis=1, inplace=True)

    return df


# Example usage
# train_data = one_hot_encode_hour(train_data, 'Time')


##############################################################################


def calculate_mean_flags(train_data, test_data, base_data, columns):
    """
    Function to calculate the mean for specified columns and flag values as higher than the mean in train and test data.

    Parameters:
    train_data (pd.DataFrame): Training data.
    test_data (pd.DataFrame): Testing data.
    base_data (pd.DataFrame): The dataset to calculate the means from.
    columns (list of tuples): List of tuples where each tuple has two values:
                              (original column name, new column prefix for flag).

    Returns:
    pd.DataFrame: Updated training and testing data with flags.
    """
    means = {}

    # Calculate means for each column in the base_data
    for col, prefix in columns:
        mean_value = base_data[col].mean()
        means[prefix] = mean_value

        # print(f'{prefix} mean: {mean_value}')

        # Add columns to flag values in train_data as higher or lower than the mean
        train_data[f'higher_{prefix}'] = train_data[col] > mean_value

        # Add columns to flag values in test_data as higher or lower than the mean
        test_data[f'higher_{prefix}'] = test_data[col] > mean_value

    return train_data, test_data


##############################################################################


def process_time_series_data(train_df, test_df, high_freq_threshold, low_freq_threshold, high_volt_threshold,
                             low_volt_threshold, time_column='Time'):
    """
    Process time series data for both train and test DataFrames by calculating deltas, flagging delta changes,
    checking ranges, and memorizing previous high deltas.

    Parameters:
    train_df (pd.DataFrame): Training time series DataFrame.
    test_df (pd.DataFrame): Testing time series DataFrame.
    high_freq_threshold (float): High threshold for frequency.
    low_freq_threshold (float): Low threshold for frequency.
    high_volt_threshold (float): High threshold for voltage.
    low_volt_threshold (float): Low threshold for voltage.
    time_column (str): Name of the time column for one-hot encoding.

    Returns:
    tuple: Processed training and testing DataFrames.
    """

    def process_single_df(df):
        # Calculate delta for frequency and voltage
        df = calculate_delta(df, 'Frequency 3E2F4FD3 (Ch Cbca Bethesda Ndosho - laboratoire)', 'Freq_delta_lab')
        df = calculate_delta(df, 'Frequency C30F2E03 (Ch Cbca Bethesda Ndosho - clinique)', 'Freq_delta_clinique')
        df = calculate_delta(df, 'Voltage 3E2F4FD3 (Ch Cbca Bethesda Ndosho - laboratoire)', 'Volt_delta_lab')
        df = calculate_delta(df, 'Voltage C30F2E03 (Ch Cbca Bethesda Ndosho - clinique)', 'Volt_delta_clinique')

        # Flagging if the delta of change is negative or positive
        df['freq_delta_negative_lab'] = df['Freq_delta_lab'] < 0
        df['freq_delta_negative_clinique'] = df['Freq_delta_clinique'] < 0
        df['volt_delta_positive_lab'] = df['Volt_delta_lab'] >= 0
        df['volt_delta_positive_clinique'] = df['Volt_delta_clinique'] >= 0

        # Checking whether the frequency and voltage are within the reasonable range
        df['freq_in_range_lab'] = (
                (df['Frequency 3E2F4FD3 (Ch Cbca Bethesda Ndosho - laboratoire)'] <= high_freq_threshold) &
                (df['Frequency 3E2F4FD3 (Ch Cbca Bethesda Ndosho - laboratoire)'] >= low_freq_threshold)
        )
        df['volt_in_range_lab'] = (
                (df['Voltage 3E2F4FD3 (Ch Cbca Bethesda Ndosho - laboratoire)'] <= high_volt_threshold) &
                (df['Voltage 3E2F4FD3 (Ch Cbca Bethesda Ndosho - laboratoire)'] >= low_volt_threshold)
        )
        df['freq_in_range_clinique'] = (
                (df['Frequency C30F2E03 (Ch Cbca Bethesda Ndosho - clinique)'] <= high_freq_threshold) &
                (df['Frequency C30F2E03 (Ch Cbca Bethesda Ndosho - clinique)'] >= low_freq_threshold)
        )
        df['volt_in_range_clinique'] = (
                (df['Voltage C30F2E03 (Ch Cbca Bethesda Ndosho - clinique)'] <= high_volt_threshold) &
                (df['Voltage C30F2E03 (Ch Cbca Bethesda Ndosho - clinique)'] >= low_volt_threshold)
        )

        # One-hot encoding for hours of the day
        df = one_hot_encode_hour(df, time_column)

        # Checking the delta of the current
        df['High_delta_freq_lab'] = abs(df['Freq_delta_lab']) > 3
        df['High_delta_freq_clinique'] = abs(df['Freq_delta_clinique']) > 3
        df['High_delta_volt_lab'] = abs(df['Volt_delta_lab']) > 10
        df['High_delta_volt_clinique'] = abs(df['Volt_delta_clinique']) > 10

        # Memorize the previous high delta
        df['Prev_High_delta_freq_lab'] = df['High_delta_freq_lab'].shift(1)
        df['Prev_High_delta_freq_clinique'] = df['High_delta_freq_clinique'].shift(1)
        df['Prev_High_delta_volt_lab'] = df['High_delta_volt_lab'].shift(1)
        df['Prev_High_delta_volt_clinique'] = df['High_delta_volt_clinique'].shift(1)

        return df

    # Process train and test DataFrames
    train_df_processed = process_single_df(train_df)
    test_df_processed = process_single_df(test_df)

    return train_df_processed, test_df_processed


# Example usage:
# Assuming train_data_ts and test_data_ts are your DataFrames
# train_data_ts, test_data_ts = process_time_series_data(train_data_ts, test_data_ts, high_freq_threshold, low_freq_threshold, high_volt_threshold, low_volt_threshold)

# Display the processed data
# display(train_data_ts)
# display(test_data_ts)


##############################################################################


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


##############################################################################


def plot_predictions(data, model_name, time_column='Time', prediction_column='prediction',
                     freq_lab_column='Frequency 3E2F4FD3 (Ch Cbca Bethesda Ndosho - laboratoire)',
                     freq_clinique_column='Frequency C30F2E03 (Ch Cbca Bethesda Ndosho - clinique)',
                     volt_lab_column='Voltage 3E2F4FD3 (Ch Cbca Bethesda Ndosho - laboratoire)',
                     volt_clinique_column='Voltage C30F2E03 (Ch Cbca Bethesda Ndosho - clinique)',
                     include_voltage=True, include_frequency=True):
    """
    Function to plot predicted Generator_ON with optional frequency and voltage data.

    Parameters:
    - data (DataFrame): The DataFrame containing the data.
    - model_name (str): The name of the model (for the plot title).
    - time_column (str): The name of the time column.
    - prediction_column (str): The name of the predicted Generator_ON column.
    - freq_lab_column (str): The column name for lab frequency.
    - freq_clinique_column (str): The column name for clinic frequency.
    - volt_lab_column (str): The column name for lab voltage.
    - volt_clinique_column (str): The column name for clinic voltage.
    - include_voltage (bool): Whether to include voltage data in the plot.
    - include_frequency (bool): Whether to include frequency data in the plot.
    """

    # Step 1: Ensure the 'Time' column is in datetime format
    data[time_column] = pd.to_datetime(data[time_column], format='%Y-%m-%d %H:%M:%S')

    # Step 2: Sort the data by 'Time' to ensure the time series is ordered correctly
    data = data.sort_values(by=time_column)

    # Create the figure and axis for predictions
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot predictions
    ax1.scatter(data[time_column], data[prediction_column], label='Prediction', color='blue', marker='o')

    # Set labels for the first axis (Prediction)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Prediction (Generator_ON)', color='black')
    ax1.set_title(f'Predicted Generator_ON with Frequency and Voltage ({model_name})')
    ax1.tick_params(axis='y', labelcolor='black')
    plt.xticks(rotation=45)

    # Plot for Frequency and Voltage using a secondary y-axis if specified
    if include_voltage or include_frequency:
        ax2 = ax1.twinx()  # Create a second y-axis

        # Plot frequency if specified
        if include_frequency:
            ax2.plot(data[time_column], data[freq_lab_column], label='Frequency Lab', color='green', linestyle='--')
            ax2.plot(data[time_column], data[freq_clinique_column], label='Frequency Clinique', color='purple',
                     linestyle='--')

        # Plot voltage if specified
        if include_voltage:
            ax2.plot(data[time_column], data[volt_lab_column], label='Voltage Lab', color='orange', linestyle='--')
            ax2.plot(data[time_column], data[volt_clinique_column], label='Voltage Clinique', color='brown',
                     linestyle='--')


        # Set the label for the secondary axis (Frequency and Voltage)
        if include_frequency and include_voltage:
            # Set the y-axis limits for frequency and voltage if either is plotted
            ax2.set_ylim(0, 275)  # Voltage and frequency common limit (0-250)
            ax2.set_yticks(range(0, 275, 25))  # Set ticks for every 50 units
            ax2.set_ylabel('Frequency (Hz) / Voltage (V)', color='black')
        elif include_frequency and not include_voltage:
            # Set the y-axis limits for frequency and voltage if either is plotted
            ax2.set_ylim(0, 60)  # Voltage and frequency common limit (0-250)
            ax2.set_yticks(range(0, 60, 10))  # Set ticks for every 50 units
            ax2.set_ylabel('Frequency (Hz)', color='black')
        else:
            # Set the y-axis limits for frequency and voltage if either is plotted
            ax2.set_ylim(0, 275)  # Voltage and frequency common limit (0-250)
            ax2.set_yticks(range(0, 275, 25))  # Set ticks for every 50 units
            ax2.set_ylabel('Voltage (V)', color='black')

        ax2.tick_params(axis='y', labelcolor='black')

    # Combine legends from both axes and place it outside the plot
    fig.legend(loc="upper center", bbox_to_anchor=(1.15, 1), bbox_transform=ax1.transAxes, borderaxespad=0)

    # Show the plot
    plt.tight_layout()
    plt.show()


##############################################################################


def plot_true_predictions_histogram(data, time_column='Time', prediction_column='prediction', minutes_interval=2):
    """
    Plots a histogram of the count of True values in the 'prediction' column per day.

    Parameters:
    - data (DataFrame): The DataFrame containing the data.
    - time_column (str): The name of the time column.
    - prediction_column (str): The name of the predicted Generator_ON column.
    - minutes_interval (int): Interval in minutes to multiply by count to get total active time in minutes.
    """

    # Step 1: Ensure the 'Time' column is in datetime format
    data[time_column] = pd.to_datetime(data[time_column], format='%Y-%m-%d %H:%M:%S')

    # Step 2: Filter the DataFrame to include only rows where predictions are True
    true_predictions = data[data[prediction_column] == True].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # Step 3: Extract the date from the 'Time' column
    true_predictions.loc[:, 'Date'] = true_predictions[time_column].dt.date

    # Step 4: Count occurrences per day, multiply by minutes_interval, and calculate hours
    daily_counts = true_predictions.groupby('Date').size().reset_index(name='Count')
    daily_counts['Minutes'] = daily_counts['Count'] * minutes_interval
    daily_counts['Hours'] = daily_counts['Minutes'] / 60

    # Step 5: Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(daily_counts['Date'], daily_counts['Minutes'], color='skyblue')
    plt.xlabel('Date')
    plt.ylabel('Minutes')
    plt.title('Daily Minutes of Generator Usage Predictions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return daily_counts


##############################################################################


# Function to calculate mean of last and future 5 rows for multiple columns, ignoring NaNs
def calculate_custom_mean(df, columns, window_size=5):
    for column in columns:
        means = []
        for i in range(len(df)):
            # Get the range of indices for last 5 and next 5 rows around the current index
            start = max(0, i - window_size)
            end = min(len(df), i + window_size + 1)

            # Calculate mean excluding NaNs
            window_data = df[column][start:end]
            mean_value = window_data.sum() / window_data.count() if window_data.count() > 0 else np.nan
            means.append(mean_value)

        # Add the result as a new column for each specified column
        df[f'{column}_windows_mean_{window_size}'] = means
    return df


# Apply the function to calculate the custom mean for multiple columns
# df = calculate_custom_mean(df, ['measurement1', 'measurement2', 'measurement3', 'measurement4'])

# Display the DataFrame with the new columns
# print(df)


##############################################################################


def check_outage_series_data(df, column, threshold=10, window_size=30):
    """
    Creates a column that counts the number of outages in the last `window_size` rows,
    where an outage is defined as a value in `column` being less than `threshold`.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to check for outages.
    threshold (float): Threshold below which a value is considered an outage.
    window_size (int): Number of previous rows to consider for counting outages.

    Returns:
    pd.DataFrame: DataFrame with an additional column indicating the count of outages.
    """
    outage_counts = []
    for i in range(len(df)):
        # Define the window range for the last `window_size` rows up to the current row
        start = max(0, i - window_size + 1)
        window_data = df[column][start:i + 1]

        # Count the number of outages in this window (values < threshold)
        outage_count = (window_data < threshold).sum()
        outage_counts.append(outage_count)

    # Add the outage count as a new column
    df[f'{column}_outage_count'] = outage_counts
    return df


# Apply the function to the DataFrame
# df = check_outage_series_data(df, 'measurement', threshold=10, window_size=30)
# Display the DataFrame with the new column
# print(df)


##############################################################################


# Function to calculate emissions
def calculate_emissions(df):

    # Constants
    NOx_lb_hp_hr = 0.024
    CO_lb_hp_hr = 0.0055
    SOx_lb_hp_hr = 0.00809
    CO2_lb_hp_hr = 1.16
    PM_lb_hp_hr = 0.0007

    # unit changes
    kwh_to_hp_hr = 1.341
    lb_to_gr = 453.592

    df['NOx Emission (g)'] = df['power (kW)'] * NOx_lb_hp_hr * kwh_to_hp_hr * lb_to_gr
    df['CO Emission (g)'] = df['power (kW)'] * CO_lb_hp_hr * kwh_to_hp_hr * lb_to_gr
    df['SOx Emission (g)'] = df['power (kW)'] * SOx_lb_hp_hr * kwh_to_hp_hr * lb_to_gr
    df['CO2 Emission (g)'] = df['power (kW)'] * CO2_lb_hp_hr * kwh_to_hp_hr * lb_to_gr
    df['PM Emission (g)'] = df['power (kW)'] * PM_lb_hp_hr * kwh_to_hp_hr * lb_to_gr
    return df


# Usage:
# test_data_2022 = calculate_emissions(test_data_2022)


##############################################################################


def generator_usage_summary(data):
    # Aggregating data
    data['Date'] = data['Time'].dt.date
    data['Month'] = data['Time'].dt.month

    # Yearly Summary
    total_minutes_year = data['prediction'].sum() * 2
    # total_hours_year = total_minutes_year / 30  # Not used but can be added if needed

    daily_usage = data.groupby('Date')['prediction'].sum() * 2
    average_daily_usage = daily_usage.mean()
    median_daily_usage = daily_usage.median()

    monthly_usage = data.groupby('Month')['prediction'].sum() * 2
    highest_month = monthly_usage.idxmax()
    highest_month_value = monthly_usage.max()
    lowest_month = monthly_usage.idxmin()
    lowest_month_value = monthly_usage.min()

    # Creating year summary DataFrame
    year_summary = pd.DataFrame({
        'Metric': [
            'Total Minutes Used (min)',
            'Average Daily Usage (min)',
            'Median Daily Usage (min)',
            'Highest Month (Total)',
            'Lowest Month (Total)'
        ],
        'Value': [
            total_minutes_year,
            average_daily_usage,
            median_daily_usage,
            f'Month {highest_month} ({highest_month_value} mins)',
            f'Month {lowest_month} ({lowest_month_value} mins)'
        ]
    })

    # Monthly Summary
    monthly_daily_avg = data.groupby('Month').apply(lambda x: x.groupby('Date')['prediction'].sum().mean())
    max_daily_usage_per_month = data.groupby('Month').apply(
        lambda x: x.groupby('Date')['prediction'].sum().max()
    )
    max_daily_usage_date_per_month = data.groupby('Month').apply(
        lambda x: x.groupby('Date')['prediction'].sum().idxmax()
    )

    monthly_summary = pd.DataFrame({
        'Month': monthly_usage.index,
        'Total Usage (min)': monthly_usage.values,
        'Average Daily Usage (min)': monthly_daily_avg.values,
        'Max Daily Usage (min)': max_daily_usage_per_month.values,
        'Max Daily Usage (Date)': max_daily_usage_date_per_month.values
    })

    return year_summary, monthly_summary


# Usage:
# year_summary, monthly_summary = power_usage_summary(data)
# display("Yearly Summary:")
# display(year_summary)
# display("\nMonthly Summary:")
# display(monthly_summary)


##############################################################################


def power_usage_summary(data):
    # Aggregating data
    data['Date'] = data['Time'].dt.date
    data['Month'] = data['Time'].dt.month

    # Yearly Summary
    total_power_year = data['power (kW)'].sum()

    daily_usage = data.groupby('Date')['power (kW)'].sum()
    average_daily_usage = daily_usage.mean()
    median_daily_usage = daily_usage.median()

    monthly_usage = data.groupby('Month')['power (kW)'].sum()
    highest_month = monthly_usage.idxmax()
    highest_month_value = monthly_usage.max()
    lowest_month = monthly_usage.idxmin()
    lowest_month_value = monthly_usage.min()

    # Creating year summary DataFrame
    year_summary = pd.DataFrame({
        'Metric': [
            'Total Power Used (kW)',
            'Average Daily Usage (kW)',
            'Median Daily Usage (kW)',
            'Highest Month (Total)',
            'Lowest Month (Total)'
        ],
        'Value': [
            total_power_year.round(2),
            average_daily_usage.round(2),
            median_daily_usage.round(2),
            f'Month {highest_month} ({highest_month_value.round(2)} kW)',
            f'Month {lowest_month} ({lowest_month_value.round(2)} kW)'
        ]
    })

    # Monthly Summary
    monthly_daily_avg = data.groupby('Month').apply(lambda x: x.groupby('Date')['power (kW)'].sum().mean())
    max_daily_usage_per_month = data.groupby('Month').apply(
        lambda x: x.groupby('Date')['power (kW)'].sum().max()
    )
    max_daily_usage_date_per_month = data.groupby('Month').apply(
        lambda x: x.groupby('Date')['power (kW)'].sum().idxmax()
    )

    monthly_summary = pd.DataFrame({
        'Month': monthly_usage.index,
        'Total Usage (kW)': monthly_usage.values.round(2),
        'Average Daily Usage (kW)': monthly_daily_avg.values.round(2),
        'Max Daily Usage (kW)': max_daily_usage_per_month.values.round(2),
        'Max Daily Usage (Date)': max_daily_usage_date_per_month.values
    })

    return year_summary, monthly_summary


# Usage:
# year_summary, monthly_summary = power_usage_summary(data)
# display("Yearly Summary:")
# display(year_summary)
# display("\nMonthly Summary:")
# display(monthly_summary)


##############################################################################


def generator_emission_summary(data):
    # List of emissions
    emissions = ['NOx', 'CO', 'SOx', 'CO2', 'PM']

    # Aggregating data
    data['Date'] = data['Time'].dt.date
    data['Month'] = data['Time'].dt.month

    for emission in emissions:
        # Power Usage Summary
        total_emission_year = data[f'{emission} Emission (g)'].sum()
        daily_emission = data.groupby('Date')[f'{emission} Emission (g)'].sum()
        average_daily_emission = daily_emission.mean()
        median_daily_emission = daily_emission.median()

        monthly_emission = data.groupby('Month')[f'{emission} Emission (g)'].sum()
        highest_month = monthly_emission.idxmax()
        highest_month_value = monthly_emission.max()
        lowest_month = monthly_emission.idxmin()
        lowest_month_value = monthly_emission.min()

        # Creating year summary DataFrame with emissions and power usage
        year_summary = pd.DataFrame({
            'Metric': [
                f'Total year {emission} Emission (g)',
                f'Average Daily {emission} Emission (g)',
                f'Median Daily {emission} Emission (g)',
                f'Highest Month {emission} Emission (Total)',
                f'Lowest Month {emission} Emission (Total)',
            ],
            'Value': [
                total_emission_year.round(2),
                average_daily_emission.round(2),
                median_daily_emission.round(2),
                f'Month {highest_month} ({highest_month_value.round(2)} g)',
                f'Month {lowest_month} ({lowest_month_value.round(2)} g)',
            ]
        })

        # Monthly Summary for emission
        monthly_daily_avg = data.groupby('Month').apply(
            lambda x: x.groupby('Date')[f'{emission} Emission (g)'].sum().mean())
        max_daily_emission_per_month = data.groupby('Month').apply(
            lambda x: x.groupby('Date')[f'{emission} Emission (g)'].sum().max()
        )
        max_daily_emission_date_per_month = data.groupby('Month').apply(
            lambda x: x.groupby('Date')[f'{emission} Emission (g)'].sum().idxmax()
        )

        monthly_summary = pd.DataFrame({
            f'Month': monthly_emission.index,
            f'Total {emission} Emission (g)': monthly_emission.values.round(2),
            f'Average Daily {emission} Emission (g)': monthly_daily_avg.values.round(2),
            f'Max Daily {emission} Emission (g)': max_daily_emission_per_month.values.round(2),
            f'Max Daily {emission} Emission (Date)': max_daily_emission_date_per_month.values
        })

        display(f"Yearly Summary {emission} emission:")
        display(year_summary)
        display(f"Monthly Summary  {emission} emission:")
        display(monthly_summary)

    # return year_summary, monthly_summary