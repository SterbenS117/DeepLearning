#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting NVIDIA CUDA and TensorFlow GPU setup for WSL Ubuntu 24.04..."

# Step 1: Update the system
echo "Updating the system..."
sudo apt update && sudo apt upgrade -y

# Step 2: Install essential packages
echo "Installing essential packages..."
sudo apt install -y build-essential wget curl git python3 python3-pip python3-dev ca-certificates gnupg lsb-release

# Step 3: Add NVIDIA CUDA repository keyring and repository
echo "Adding NVIDIA CUDA repository key and repository..."

# Download the keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-archive-keyring.gpg

# Move the keyring to trusted.gpg.d
sudo mv cuda-archive-keyring.gpg /usr/share/keyrings/cuda-archive-keyring.gpg

# Add the repository with the keyring
echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda-wsl.list

# Step 4: Update package lists
sudo apt update

# Step 5: Install CUDA Toolkit
echo "Installing CUDA Toolkit..."
sudo apt install -y cuda-toolkit-12-3

# Step 6: Add CUDA to PATH and LD_LIBRARY_PATH
echo "Configuring environment variables..."
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Step 7: Install cuDNN
echo "Installing cuDNN..."

CUDNN_TAR_FILE="cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz"

# Download cuDNN (you must have an NVIDIA Developer account to download this file)
wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.9.7/${CUDNN_TAR_FILE} -O ${CUDNN_TAR_FILE}

# Extract and copy cuDNN files
tar -xf ${CUDNN_TAR_FILE}
CUDNN_DIR="cudnn-linux-x86_64-8.9.7.29_cuda12-archive"

sudo cp -P ${CUDNN_DIR}/include/* /usr/local/cuda/include
sudo cp -P ${CUDNN_DIR}/lib/* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

# Step 8: Install TensorFlow with GPU support
echo "Installing TensorFlow with GPU support..."
pip3 install --upgrade pip
pip3 install tensorflow tensorflow-gpu

# Step 9: Verify installation
echo "Verifying installation..."
nvidia-smi
python3 -c "import tensorflow as tf; print('TensorFlow GPU:', tf.config.list_physical_devices('GPU'))"

echo "Installation complete! Please restart your WSL instance to apply changes."
