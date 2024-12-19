import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Enable GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print(f"Using GPU: {gpus[0]}")
    except RuntimeError as e:
        print(e)


def divide_dataframe(df, n):
    """
    Divides a DataFrame into n equal (or near-equal) parts.

    Args:
        df (pd.DataFrame): The DataFrame to be divided.
        n (int): The number of parts to divide the DataFrame into.

    Returns:
        list: A list containing the divided DataFrames.
    """
    if n <= 0:
        raise ValueError("The number of parts 'n' must be a positive integer.")

    # Determine the approximate size of each part
    chunk_size = len(df) // n
    remainder = len(df) % n

    # Create a list of sizes for each chunk, distributing the remainder
    sizes = [chunk_size + 1 if i < remainder else chunk_size for i in range(n)]

    # Divide the DataFrame based on the calculated sizes
    indices = np.cumsum([0] + sizes)
    return [df.iloc[indices[i]:indices[i + 1]] for i in range(n)]

def dl_function(train, test):
    # Standardize features
    # Split features and labels
    X_train = train[
        ['Clay', 'Sand', 'Silt', 'Elevation', 'Aspect', 'Slope', 'MODIS', 'LAI', 'ALB', 'Temp', "Date"]]
    y_train = train['Smerge']
    X_test = test[['Clay', 'Sand', 'Silt', 'Elevation', 'Aspect', 'Slope', 'MODIS', 'LAI', 'ALB', 'Temp', "Date"]]
    y_test = test['Smerge']
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert NumPy arrays to TensorFlow datasets for efficient loading
    def preprocess_data(features, labels):
        features = tf.cast(features, tf.float32)
        return features, labels

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = (train_dataset
                     .shuffle(buffer_size=100000)
                     .map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
                     .batch(256)
                     .cache()  # Cache to memory to alleviate CPU bottlenecks
                     .prefetch(tf.data.AUTOTUNE))

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = (test_dataset
                    .map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(256)
                    .cache()
                    .prefetch(tf.data.AUTOTUNE))

    # Define the model
    def build_model():
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)  # Output layer for regression
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    # Build and train the model
    model = build_model()
    model.fit(train_dataset, epochs=50, verbose=2)

    # Predict on the test dataset
    pred = model.predict(test_dataset)
    del model
    return pred



# Load or distribute your data here
#home = r'E:\BigRun'
train_file = '/mnt/e/BigRun/BigRunWS_V5_T_500_train_main.csv'
test_file = '/mnt/e/BigRun/BigRunWS_V5_T_500_test_main.csv'
n = 6
train_data_full = pd.read_csv(train_file, usecols=['PageName', 'Clay', 'Sand', 'Silt', 'Elevation', 'Slope', 'Aspect', 'MODIS', 'Smerge', 'Date', 'LAI', 'ALB', 'Temp'], engine='pyarrow')
test_data_full = pd.read_csv(test_file, usecols=['PageName', 'Clay', 'Sand', 'Silt', 'Elevation', 'Slope', 'Aspect', 'MODIS', 'Smerge', 'Date', 'LAI', 'ALB', 'Temp', 'AHRR'], engine='pyarrow')
ahrr = test_data_full['AHRR']
smerge = test_data_full['Smerge']
date = test_data_full['Date']
pagename = test_data_full['PageName']
train_data = train_data_full[['Clay', 'Sand', 'Silt', 'Elevation', 'Aspect', 'Slope', 'MODIS', 'Smerge', 'Date', 'LAI', 'ALB', 'Temp']].dropna()
test_data = test_data_full[['Clay', 'Sand', 'Silt', 'Elevation', 'Aspect', 'Slope', 'MODIS', 'Smerge', 'Date', 'LAI', 'ALB', 'Temp']].dropna()

# Convert Date to a numerical value
test_data['Date'] = pd.to_datetime(test_data['Date'], format="%Y-%m-%d").astype(int)
train_data['Date'] = pd.to_datetime(train_data['Date'], format="%Y-%m-%d").astype(int)

test_list = divide_dataframe(test_data, n)
train_list = divide_dataframe(train_data, n)

del train_data

pred_out = pd.DataFrame(columns=['ML_'])
for i in range(n):
    pred_temp = pd.DataFrame(columns=['ML_'])
    j = dl_function(train_list[i], test_list[i])
    j = j.reshape(-1)
    pred_temp['ML_'] = j
    pred_out = pd.concat([pred_out, pred_temp], ignore_index=True)

# Save predictions
test_data['ML_'] = pred_out['ML_']
test_data['AHRR'] = ahrr
test_data['SMERGE'] = smerge
test_data['Date'] = date
test_data['PageName'] = pagename
test_data.to_csv("/mnt/e/BigRun/TFDL_BigRunWS_V5_T_500.csv", index=False)
