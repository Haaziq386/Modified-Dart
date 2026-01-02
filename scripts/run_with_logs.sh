#!/bin/bash

# run_with_logs.sh
# Usage: sh scripts/run_with_logs.sh <dataset_name>
# Example: sh scripts/run_with_logs.sh ETTh1
# This will run both pretrain and finetune scripts with logging to outputs/logs/

set -e

if [ -z "$1" ]; then
    echo "Usage: sh scripts/run_with_logs.sh <dataset_name>"
    echo "Example: sh scripts/run_with_logs.sh ETTh1"
    exit 1
fi

DATASET=$1
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./outputs/logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

PRETRAIN_LOG="$LOG_DIR/pretrain_${DATASET}_${TIMESTAMP}.log"
FINETUNE_LOG="$LOG_DIR/finetune_${DATASET}_${TIMESTAMP}.log"

echo "=========================================="
echo "Starting training pipeline for $DATASET"
echo "Timestamp: $TIMESTAMP"
echo "=========================================="
echo ""

# Run pretrain with logging
echo ">>> Running PRETRAIN for $DATASET"
echo "    Log file: $PRETRAIN_LOG"
if bash "scripts/pretrain/${DATASET}.sh" 2>&1 | tee "$PRETRAIN_LOG"; then
    echo ""
    echo ">>> PRETRAIN completed successfully"
    echo ""
else
    echo ""
    echo ">>> PRETRAIN failed! Check log file: $PRETRAIN_LOG"
    exit 1
fi

# Run finetune with logging
echo ">>> Running FINETUNE for $DATASET"
echo "    Log file: $FINETUNE_LOG"
if bash "scripts/finetune/${DATASET}.sh" 2>&1 | tee "$FINETUNE_LOG"; then
    echo ""
    echo ">>> FINETUNE completed successfully"
    echo ""
else
    echo ""
    echo ">>> FINETUNE failed! Check log file: $FINETUNE_LOG"
    exit 1
fi

echo "=========================================="
echo "Training pipeline completed successfully!"
echo "=========================================="
echo "Pretrain log: $PRETRAIN_LOG"
echo "Finetune log: $FINETUNE_LOG"
