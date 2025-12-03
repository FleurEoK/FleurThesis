#!/bin/bash
# Run diagnostic script interactively in terminal
# Usage: bash run_diagnose.sh

echo "Starting diagnostic script..."
echo "Output will appear in real-time in this terminal"
echo ""

# Navigate to SemAIM directory
# cd /home/20204130/SemAIM/SemAIM-master

# Load required modules
# source /home/20204130/Falcon/myvenv311

# Run diagnostic with unbuffered output
python -u ~/SemAIM/SemAIM-master/diagnosis_error.py

echo ""
echo "Diagnostic complete!"