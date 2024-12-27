import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

major_chunk = str(3)
#major_chunk = str(sys.argv[1])
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

def dl_function(train, test):
    # Split features and labels
    X_train = train[
        ['Clay', 'Sand', 'Silt', 'Elevation', 'Aspect', 'Slope', 'MODIS', 'LAI', 'ALB', 'Temp', "Date"]]
    y_train = train['Smerge']
    X_test = test[['Clay', 'Sand', 'Silt', 'Elevation', 'Aspect', 'Slope', 'MODIS', 'LAI', 'ALB', 'Temp', "Date"]]
    y_test = test['Smerge']
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
                     .batch(16)
                     .cache()  # Cache to memory to alleviate CPU bottlenecks
                     .prefetch(tf.data.AUTOTUNE))

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = (test_dataset
                    .map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(16)
                    .cache()
                    .prefetch(tf.data.AUTOTUNE))

    # Define the model
    def build_model():
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(X_train.shape[1],)),
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
    return pred

# Load or distribute your data here
#home = r'E:\BigRun'
train_file = '/mnt/e/BigRun/train/BigRunWS_V5_T_500_train_part_'+major_chunk+'.csv'
test_file = '/mnt/e/BigRun/test/BigRunWS_V5_T_500_test_part_'+major_chunk+'.csv'

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


pred_out = dl_function(train_data, test_data)


# Save predictions
test_data['ML_'] = pred_out
test_data['AHRR'] = ahrr
test_data['SMERGE'] = smerge
test_data['Date'] = date
test_data['PageName'] = pagename
test_data.to_csv("/mnt/e/BigRun/TFDL_BigRunWS_V5_TS_500_part_"+major_chunk+".csv", index=False)