#!/bin/bash

# Run evaluation script
# Usage: ./scripts/run_evaluation.sh <base_dir>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <base_dir>"
    exit 1
fi

BASE_DIR=$1

echo "Running evaluation with BASE_DIR=$BASE_DIR"

python src/evaluation.py --base_dir "$BASE_DIR"