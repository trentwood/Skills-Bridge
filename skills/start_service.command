#!/bin/bash

# Skill Extractor Web Service Launcher
# Double-click this file to start the service

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to that directory
cd "$DIR"

# Print startup message
echo "================================================"
echo "Starting Skill Extractor Web Service"
echo "================================================"
echo "Directory: $DIR"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Start the service
echo "Starting service on http://localhost:8000"
echo ""
echo "API Documentation: http://localhost:8000/docs"
echo "Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the service"
echo "================================================"
echo ""

# Run the service
python skill_extractor_service.py

# Keep terminal open if there's an error
if [ $? -ne 0 ]; then
    echo ""
    echo "Service stopped with an error. Press any key to close..."
    read -n 1
fi
