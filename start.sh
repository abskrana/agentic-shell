#!/bin/bash

# Agentic Shell - Start Script

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  Starting Agentic Shell"
echo "=========================================="
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found"
    echo "Please run ./install.sh first to set up the environment"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d .venv ]; then
    echo "Error: Virtual environment not found"
    echo "Please run ./install.sh first to set up the environment"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Start the application
echo "Starting server..."
echo ""
uv run main.py