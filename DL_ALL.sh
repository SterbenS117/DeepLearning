#!/bin/bash

# First group of 3
python3 /mnt/c/Users/Administrator/PycharmProjects/DeepLearning/deep_learning_optimized_pipeline.py 1 &
wait
python3 /mnt/c/Users/Administrator/PycharmProjects/DeepLearning/deep_learning_optimized_pipeline.py 2 &
wait

# Second group of 3
python3 /mnt/c/Users/Administrator/PycharmProjects/DeepLearning/deep_learning_optimized_pipeline.py 3 &
wait
python3 /mnt/c/Users/Administrator/PycharmProjects/DeepLearning/deep_learning_optimized_pipeline.py 5 &
wait

python3 /mnt/c/Users/Administrator/PycharmProjects/DeepLearning/deep_learning_optimized_pipeline.py 4 &
wait
python3 /mnt/c/Users/Administrator/PycharmProjects/DeepLearning/deep_learning_optimized_pipeline.py 6 &
wait

# Fourth group of 2
python3 /mnt/c/Users/Administrator/PycharmProjects/DeepLearning/deep_learning_optimized_pipeline.py 7 &
wait
python3 /mnt/c/Users/Administrator/PycharmProjects/DeepLearning/deep_learning_optimized_pipeline.py 8 &
wait

python3 /mnt/c/Users/Administrator/PycharmProjects/DeepLearning/deep_learning_optimized_pipeline.py 9 &
wait
python3 /mnt/c/Users/Administrator/PycharmProjects/DeepLearning/deep_learning_optimized_pipeline.py 10 &
wait