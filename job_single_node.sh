#!/bin/bash -l
### Request a single task using one core on one node for 5 minutes in the batch queue
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --time=0-01:00:00
#SBATCH -p batch
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --job-name="ML_task1"

# Activate the virtual environment
micromamba activate project

# Define the number of repetitions
NUM_RUNS=1
NUM_CORES=64

# Define the path to the Python script
SCRIPT_PATH="test_single_node.py"

# Define the log file for execution times
TIME_LOG="./output/joblib/time_log_$NUM_CORES.txt"
# ensure directory exists
mkdir -p ./output/joblib

# Loop to run the script multiple times
for i in $(seq 1 $NUM_RUNS); do
    echo "Starting run $i..."
    # Measure the time and write to the log
    { time python3 "$SCRIPT_PATH" $NUM_CORES > "./output/joblib/output_joblib_$NUM_CORES.log" 2> "./output/joblib/error_joblib_$NUM_CORES.log"; } 2>> "$TIME_LOG"
    echo "Run $i completed."
    # Add a separator in the log file for clarity
    echo "----- End of run $i -----" >> "$TIME_LOG"
done
