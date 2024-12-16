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
                     .shuffle(buffer_size=10000)
                     .map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
                     .batch(32)
                     .cache()  # Cache to memory to alleviate CPU bottlenecks
                     .prefetch(tf.data.AUTOTUNE))

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = (test_dataset
                    .map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(32)
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
