import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
#from tensorflow.keras import mixed_precision

# Enable mixed precision training
#mixed_precision.set_global_policy('mixed_float16')

# Set GPU configuration to use NVIDIA CUDA devices
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print(f"Using GPU: {gpus[0]}")
    except RuntimeError as e:
        print(e)

# Load or distribute your data here
home = r'E:\BigRun'
train_file = '/mnt/e/BigRun/BigRunWS_V5_TS_500_train_part_1.csv'
test_file = '/mnt/e/BigRun/BigRunWS_V5_TS_500_test_part_1.csv'

train_data_full = pd.read_csv(train_file, usecols=['PageName', 'Clay', 'Sand', 'Silt', 'Elevation', 'Slope', 'Aspect', 'MODIS', 'Smerge', 'Date', 'LAI', 'ALB', 'Temp'], engine='pyarrow')
test_data_full = pd.read_csv(test_file, usecols=['PageName', 'Clay', 'Sand', 'Silt', 'Elevation', 'Slope', 'Aspect', 'MODIS', 'Smerge', 'Date', 'LAI', 'ALB', 'Temp', 'AHRR'], engine='pyarrow')

train_data = train_data_full[['Clay', 'Sand', 'Silt', 'Elevation', 'Aspect', 'Slope', 'MODIS', 'Smerge', 'Date', 'LAI', 'ALB', 'Temp']].dropna()
test_data = test_data_full[['Clay', 'Sand', 'Silt', 'Elevation', 'Aspect', 'Slope', 'MODIS', 'Smerge', 'Date', 'LAI', 'ALB', 'Temp']].dropna()

# Convert Date to a numerical value
test_data['Date'] = pd.to_datetime(test_data['Date'], format="%Y-%m-%d").astype(int)
train_data['Date'] = pd.to_datetime(train_data['Date'], format="%Y-%m-%d").astype(int)

# Split features and labels
X_train = train_data[['Clay', 'Sand', 'Silt', 'Elevation', 'Aspect', 'Slope', 'MODIS', 'LAI', 'ALB', 'Temp', "Date"]]
y_train = train_data['Smerge']
X_test = test_data[['Clay', 'Sand', 'Silt', 'Elevation', 'Aspect', 'Slope', 'MODIS', 'LAI', 'ALB', 'Temp', "Date"]]
y_test = test_data['Smerge']

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert NumPy arrays to TensorFlow datasets for efficient loading
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Define the model with mixed precision
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, dtype='float32')  # Output layer for regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Build and train the model
model = build_model()

# Enable TensorBoard profiling
log_dir = "/mnt/c/Users/asanchez2415/PycharmProjects/Deep_Learning2025"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch='500,520')

model.fit(train_dataset, epochs=50, verbose=2, callbacks=[tensorboard_callback])

# Predict on the test dataset
pred = model.predict(test_dataset)

# Save predictions
test_data['ML_'] = pred
test_data.to_csv("/mnt/e/BigRun/TFDL_BigRunWS_V5_TS_500.csv", index=False)
