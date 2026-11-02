#!/usr/bin/env bash
set -e

# Ensure a Conda environment is activated
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "Error: Activate the openotfs Conda environment first."
    exit 1
fi

echo "Installing OpenNTN 1.0.0..."

# Download the installer for the compatible OpenNTN version
curl -fL \
    https://raw.githubusercontent.com/ant-uni-bremen/OpenNTN/v1.0.0/install.sh \
    -o /tmp/openotfs_install_openntn.sh

# Run the installer
bash /tmp/openotfs_install_openntn.sh

# Verify the installation
python -c 'from sionna.phy.channel.tr38811 import DenseUrban'

echo "OpenNTN installation successful!"