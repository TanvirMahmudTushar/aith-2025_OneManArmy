#!/bin/bash

# Check Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Python is not installed. Please install Python 3 first."
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv 2>/dev/null || python -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment (Linux/macOS)..."
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    echo "Activating virtual environment (Windows Git Bash)..."
    source venv/Scripts/activate
else
    echo "Cannot find virtual environment activation script!"
    exit 1
fi

# Upgrade pip and install dependencies
echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete!"
echo "To run inference, use: bash run.sh"
