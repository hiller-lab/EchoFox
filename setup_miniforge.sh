#!/bin/bash

# EchoFox Miniforge Environment Setup Script
# This script sets up a Miniforge conda environment for the EchoFox project

set -e  # Exit on error

ENV_NAME="echofox"
PYTHON_VERSION="3.14"

echo "Setting up Miniforge environment for EchoFox..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Miniforge from: https://github.com/conda-forge/miniforge"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove it and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Aborting setup."
        exit 0
    fi
fi

# Create new conda environment
echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Install the package in development mode with all dependencies
if [ -f "pyproject.toml" ] || [ -f "setup.py" ]; then
    echo "Installing EchoFox with all dependencies..."
    pip install --upgrade pip
    pip install -e ".[all]"
fi

echo ""
echo "✓ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To deactivate, run:"
echo "  conda deactivate"
