#!/bin/bash

echo "=========================================="
echo "Job Architecture Service Startup"
echo "=========================================="
echo ""


# Check if required files exist
REQUIRED_FILES=(
    "job_graph.json"
    "normalizer_data.pkl"
    "statistics.json"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: Required file not found: $file"
        echo ""
        echo "Please run the Jupyter notebook to generate all data files."
        exit 1
    fi
done

echo "✓ All data files found"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

echo "Starting Job Architecture API Service..."
echo ""
echo "Service will be available at:"
echo "  http://localhost:5001"
echo ""
echo "Press Ctrl+C to stop the service"
echo ""
echo "=========================================="
echo ""

python job_architecture_service.py
