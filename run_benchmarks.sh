#!/bin/bash -l
#SBATCH --job-name=benchmark
#SBATCH --output=benchmark.out
#SBATCH --error=benchmark.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=02:00:00
#SBATCH --partition=batch

module load lang/Python/3.8.6-GCCcore-10.2.0

# Put your environment
micromamba activate forecast_venv


# Put your project path
cd /scratch/users/dferrario/ForecastersEnsemble

for np in 1 2 4 8 16 32 64 128
do
    echo "Running with $np processes"
    mpiexec -n $np python benchmark.py
done
