#!/bin/bash

# Skills-Bridge Service Startup Script
# Starts both Job Architecture and Skill Extraction services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "Starting Skills-Bridge Services"
echo "=============================================="

# Check for required data files
echo "Checking data files..."

JOB_DATA="$PROJECT_ROOT/data/job_architecture"
SKILLS_DATA="$PROJECT_ROOT/data/skills"

if [ ! -f "$JOB_DATA/job_graph.json" ]; then
    echo "ERROR: Job architecture data not found at $JOB_DATA"
    echo "Please run the job_architecture_with_soc.ipynb notebook first."
    exit 1
fi

if [ ! -f "$SKILLS_DATA/skill_taxonomy.parquet" ]; then
    echo "ERROR: Skills data not found at $SKILLS_DATA"
    echo "Please run the Skill Taxonomy and Extraction.ipynb notebook first."
    exit 1
fi

echo "Data files found!"

# Start Job Architecture Service
echo ""
echo "Starting Job Architecture Service on port 5001..."
cd "$PROJECT_ROOT"
python src/job_architecture/service.py &
JOB_PID=$!

# Wait a moment for the first service to start
sleep 2

# Start Skill Extraction Service
echo ""
echo "Starting Skill Extraction Service on port 8000..."
uvicorn src.skill_extraction.service:app --host 0.0.0.0 --port 8000 &
SKILLS_PID=$!

echo ""
echo "=============================================="
echo "Services started!"
echo "=============================================="
echo "Job Architecture: http://localhost:5001"
echo "Skill Extraction: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop all services"
echo "=============================================="

# Wait for interrupt
trap "kill $JOB_PID $SKILLS_PID 2>/dev/null" EXIT
wait
