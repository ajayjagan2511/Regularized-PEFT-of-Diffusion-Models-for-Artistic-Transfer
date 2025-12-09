#!/bin/bash

# Preprocess images script
# Usage: ./scripts/run_preprocess.sh <image_dir>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <image_dir>"
    exit 1
fi

IMAGE_DIR=$1
TARGET_SIZE=1024

echo "Running preprocess_images.py with IMAGE_DIR=$IMAGE_DIR and TARGET_SIZE=$TARGET_SIZE"

python src/preprocess_images.py --image_dir "$IMAGE_DIR" --target_size "$TARGET_SIZE"