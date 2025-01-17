import os
import sys
import math
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt  # Keras Tuner for hyperparameter tuning
tf.config.optimizer.set_jit(False)

# cpus = tf.config.experimental.list_physical_devices('CPU')
# tf.config.experimental.set_visible_devices(cpus[0], 'CPU')
#
# os.environ["OMP_NUM_THREADS"] = "56"
# os.environ["TF_NUM_INTEROP_THREADS"] = "56"
# os.environ["TF_NUM_INTRAOP_THREADS"] = "56"
#
# tf.config.threading.set_intra_op_parallelism_threads(16)
# tf.config.threading.set_inter_op_parallelism_threads(16)
#
# print("Intra-op threads:", tf.config.threading.get_intra_op_parallelism_threads())


major_chunk = str(4)

#Enable GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print(f"Using GPU: {gpus[0]}")
    except RuntimeError as e:
        print(e)


# Define model builder function for Keras Tuner
def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(shape=(11,)))  # Fixed input shape

    # Add hidden layers with hyperparameters
    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(tf.keras.layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
            activation=hp.Choice('activation', ['relu', 'tanh'])
        ))

    # Output layer
    model.add(tf.keras.layers.Dense(1))

    # Compile the model with a tunable learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [1e-3, 1e-4, 1e-5])),
        loss='mean_squared_error',
        metrics=['mae']
    )
    return model

def dl_function(train, test):
    # Split features and labels
    X_train = train[['Clay', 'Sand', 'Silt', 'Elevation', 'Aspect', 'Slope', 'MODIS', 'LAI', 'ALB', 'Temp', "Date"]]
    y_train = train['Smerge']
    X_test = test[['Clay', 'Sand', 'Silt', 'Elevation', 'Aspect', 'Slope', 'MODIS', 'LAI', 'ALB', 'Temp', "Date"]]
    y_test = test['Smerge']

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert NumPy arrays to TensorFlow datasets
    def preprocess_data(features, labels):
        features = tf.cast(features, tf.float32)
        return features, labels

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    n = len(train_dataset)
    top_90 = math.floor(0.9 * n)

    train_dataset_val = (train_dataset.skip(top_90)
                     .map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
                     .batch(128)
                     .cache()
                     .prefetch(tf.data.AUTOTUNE))
    train_dataset = (train_dataset.take(top_90)
                     .shuffle(buffer_size=10000)
                     .map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
                     .batch(128)
                     .cache()
                     .prefetch(tf.data.AUTOTUNE))

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = (test_dataset
                    .map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(128)
                    .cache()
                    .prefetch(tf.data.AUTOTUNE))

    # Initialize Keras Tuner
    tuner = kt.RandomSearch(
        build_model,
        objective='val_mae',
        max_trials=25,  # Number of hyperparameter combinations to try
        executions_per_trial=1,
        directory='hyperparameter_tuning_'+major_chunk,
        project_name='dl_pipeline_tuning'
    )

    tuner.search(train_dataset, epochs=12, validation_data=train_dataset_val, verbose=2)

    # Retrieve the best model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    model.fit(train_dataset, epochs=45, verbose=2)

    # Predict on the test dataset
    pred = model.predict(test_dataset)
    return pred

# Load or distribute your data here
# train_file = '/mnt/e/BigRun/train/BigRunWS_V5_T_500_train_part_'+major_chunk+'.csv'
# test_file = '/mnt/e/BigRun/test/BigRunWS_V5_T_500_test_part_'+major_chunk+'.csv'
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
