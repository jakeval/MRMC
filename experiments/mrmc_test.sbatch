#! /bin/bash

#SBATCH --job-name=mrmc_experiment_prototype
#SBATCH --output=result-%A_%a.out
#SBATCH --error=result-%A_%a.err
#SBATCH --time=0-00:05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=defq

export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

module load shared python/3.8.11

source $1

python $2 --n_trials 30 --max_trials 5 --n_procs 1