import tensorflow as tf

# Reset memory stats
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.reset_memory_stats(gpu)
