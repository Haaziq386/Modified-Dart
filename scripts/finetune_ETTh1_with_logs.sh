#!/bin/bash

# Wrapper for finetune/ETTh1.sh with logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./outputs/logs/finetune_ETTh1_${TIMESTAMP}.log"

mkdir -p ./outputs/logs

echo "Starting finetune for ETTh1 ($(date))"
echo "Log file: $LOG_FILE"
bash scripts/finetune/ETTh1.sh 2>&1 | tee "$LOG_FILE"
exit_code=${PIPESTATUS[0]}

if [ $exit_code -eq 0 ]; then
    echo "Finetune completed successfully"
else
    echo "Finetune failed with exit code $exit_code"
fi

exit $exit_code
