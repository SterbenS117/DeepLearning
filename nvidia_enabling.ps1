# PowerShell Script to Install NVIDIA GPU Driver, CUDA Toolkit, cuDNN, and TensorFlow on Windows

# Define URLs for the required installers
$driverUrl = "https://us.download.nvidia.com/Windows/531.79/531.79-desktop-win10-win11-64bit-international-dch-whql.exe"
$cudaUrl = "https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_528.33_windows.exe"
$cudnnUrl = "https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.9.1/local_installers/11.8/cudnn-windows-x86_64-8.9.1.23_cuda11-archive.zip"

# Define installation paths
$installPath = "C:\Installers"
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"

# Create directory for installers
New-Item -ItemType Directory -Force -Path $installPath

# Function to download files
function Download-File {
    param (
        [string]$url,
        [string]$output
    )
    Invoke-WebRequest -Uri $url -OutFile $output
}

# Download NVIDIA GPU Driver
$driverInstaller = "$installPath\NVIDIA_Driver.exe"
Download-File -url $driverUrl -output $driverInstaller

# Download CUDA Toolkit
$cudaInstaller = "$installPath\CUDA_Toolkit.exe"
Download-File -url $cudaUrl -output $cudaInstaller

# Download cuDNN
$cudnnArchive = "$installPath\cuDNN.zip"
Download-File -url $cudnnUrl -output $cudnnArchive

# Install NVIDIA GPU Driver
Start-Process -FilePath $driverInstaller -ArgumentList "-s" -Wait

# Install CUDA Toolkit
Start-Process -FilePath $cudaInstaller -ArgumentList "-s" -Wait

# Extract cuDNN files
Expand-Archive -Path $cudnnArchive -DestinationPath $installPath

# Copy cuDNN files to CUDA directory
Copy-Item -Path "$installPath\cuda\bin\*" -Destination "$cudaPath\bin" -Force
Copy-Item -Path "$installPath\cuda\include\*" -Destination "$cudaPath\include" -Force
Copy-Item -Path "$installPath\cuda\lib\x64\*" -Destination "$cudaPath\lib\x64" -Force

# Set environment variables
[System.Environment]::SetEnvironmentVariable("CUDA_PATH", $cudaPath, [System.EnvironmentVariableTarget]::Machine)
$env:Path += ";$cudaPath\bin;$cudaPath\lib\x64"
[System.Environment]::SetEnvironmentVariable("Path", $env:Path, [System.EnvironmentVariableTarget]::Machine)

# Install TensorFlow with GPU support
pip install tensorflow

# Clean up installers
Remove-Item -Path $installPath -Recurse -Force

Write-Host "Installation complete. Please restart your computer to apply the changes."
