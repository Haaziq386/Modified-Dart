#!/bin/bash

# Wrapper for pretrain/ETTh1.sh with logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./outputs/logs/pretrain_ETTh1_${TIMESTAMP}.log"

mkdir -p ./outputs/logs

echo "Starting pretrain for ETTh1 ($(date))"
echo "Log file: $LOG_FILE"
bash scripts/pretrain/ETTh1.sh 2>&1 | tee "$LOG_FILE"
exit_code=${PIPESTATUS[0]}

if [ $exit_code -eq 0 ]; then
    echo "Pretrain completed successfully"
else
    echo "Pretrain failed with exit code $exit_code"
fi

exit $exit_code
