#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting NVIDIA and TensorFlow GPU setup for WSL Ubuntu 24.04..."

# Step 1: Update the system
echo "Updating the system..."
sudo apt update && sudo apt upgrade -y

# Step 2: Install essential packages
echo "Installing essential packages..."
sudo apt install -y build-essential wget curl git python3 python3-pip python3-dev

# Step 3: Install NVIDIA drivers for WSL
echo "Installing NVIDIA drivers..."
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda-repo-wsl-ubuntu-12-3-local_12.3.2-535.104.05-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-3-local_12.3.2-535.104.05-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub
sudo apt update
sudo apt install -y cuda

# Step 4: Add CUDA to PATH and LD_LIBRARY_PATH
echo "Adding CUDA to PATH and LD_LIBRARY_PATH..."
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Step 5: Install cuDNN
echo "Installing cuDNN..."
CUDNN_VERSION="8.9.2.26"
wget https://developer.nvidia.com/compute/cudnn/secure/8.9.2/local_installers/11.8/cudnn-linux-x86_64-8.9.2.26_cuda11-archive.tar.xz -O cudnn.tar.xz
tar -xf cudnn.tar.xz
sudo cp -P cudnn-linux-x86_64-8.9.2.26_cuda11-archive/include/* /usr/local/cuda/include
sudo cp -P cudnn-linux-x86_64-8.9.2.26_cuda11-archive/lib/* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

# Step 6: Install TensorFlow
echo "Installing TensorFlow..."
pip3 install --upgrade pip
pip3 install tensorflow tensorflow-gpu

# Step 7: Verify installation
echo "Verifying installation..."
nvidia-smi
python3 -c "import tensorflow as tf; print('TensorFlow GPU:', tf.config.list_physical_devices('GPU'))"

echo "Installation complete! Reboot your system to apply changes."
