import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


# # Set GPU configuration to use NVIDIA CUDA devices
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#     except RuntimeError as e:
#         print(e)
# print(gpus)

for i in range(6):
    print(i)