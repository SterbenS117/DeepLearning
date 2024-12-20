import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("Running on CPU")
# # Enable GPU configuration
#cpus = tf.config.experimental.list_physical_devices('CPU')
# if cpus:
#     try:
#         for cpu in cpus:
#             tf.config.experimental.set_memory_growth(cpu, True)
#             tf.config.experimental.set_visible_devices(cpus[0], 'CPU')
#         print(f"Using GPU: {cpus[0]}")
#     except RuntimeError as e:
#         print(e)

# Load or distribute your data here
#home = r'E:\BigRun'
train_file = '/mnt/e/BigRun/train/BigRunWS_V5_T_500_train_part_2.csv'
test_file = '/mnt/e/BigRun/test/BigRunWS_V5_T_500_test_part_2.csv'

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
def preprocess_data(features, labels):
    features = tf.cast(features, tf.float32)
    return features, labels

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = (train_dataset
                 .shuffle(buffer_size=10000)
                 .map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
                 .batch(64)
                 .cache()  # Cache to memory to alleviate CPU bottlenecks
                 .prefetch(tf.data.AUTOTUNE))

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = (test_dataset
                .map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(64)
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

# Save predictions
test_data['ML_'] = pred
test_data['AHRR'] = ahrr
test_data['SMERGE'] = smerge
test_data['Date'] = date
test_data['PageName'] = pagename
test_data.to_csv("/mnt/e/BigRun/TFDL_BigRunWS_V5_TS_500_part_2.csv", index=False)
