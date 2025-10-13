#!/usr/bin/env bash
set -euo pipefail

# 1) Make sure conda’s shell functions are available:
eval "$(conda shell.bash hook)"
conda activate visprog

# 2) Change to the directory of interest:
cd /data3/zihanwa3/Capstone-DSR/Appendix/segment-anything-2
#!/bin/bash

# Usage:
#   ./parallel_mask.sh /path/to/data
#
# This will automatically:
#   1) Append "_img" to your input directory (e.g., DATA_PATH="/path/to/data_img").
#   2) Detect how many GPUs are available (GPU_COUNT).
#   3) Enumerate each subdirectory of DATA_PATH.
#   4) Round-robin assign each subdirectory to a GPU.
#   5) Run the Python script in parallel on each GPU.

DATA_PATH="${1}_img"

# Count how many GPUs you have
GPU_COUNT=$(nvidia-smi -L | wc -l)

# Gather all subdirectories (one level deep) under DATA_PATH
folders=($(find "$DATA_PATH" -maxdepth 1 -mindepth 1 -type d))

echo "Found ${#folders[@]} subdirectories in $DATA_PATH"
echo "Detected $GPU_COUNT GPUs"

# Loop over all subdirectories, launching a separate process for each
for i in "${!folders[@]}"; do
    folder="${folders[$i]}"
    seq=$(basename "$folder")
    
    # Round-robin assignment of GPU by index modulo GPU_COUNT
    GPU_ID=$(( i % GPU_COUNT ))
    
    echo "Launching job for '$folder' on GPU $GPU_ID ..."
    
    # Run the Python script in the background
    CUDA_VISIBLE_DEVICES=$GPU_ID \
      python custom_mask.py \
        --seq "$seq" \
        --text_prompt "person" \
        --video_dir "$DATA_PATH" &

    if [[ "$DATA_PATH" == *door* ]]; then
        echo "'door' found in DATA_PATH: $DATA_PATH. Running twice..."
        CUDA_VISIBLE_DEVICES=$GPU_ID \
          python custom_mask.py \
            --seq "$seq" \
            --text_prompt "door" \
            --video_dir "$DATA_PATH" &
    fi

    if [[ "$DATA_PATH" == *box* ]]; then
        echo "'door' found in DATA_PATH: $DATA_PATH. Running twice..."
        CUDA_VISIBLE_DEVICES=$GPU_ID \
          python custom_mask.py \
            --seq "$seq" \
            --text_prompt "box" \
            --video_dir "$DATA_PATH" &
    fi

done

# Wait for all background processes to finish before exiting
wait
echo "All parallel jobs finished."
