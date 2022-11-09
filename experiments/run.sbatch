#! /bin/bash

#SBATCH --job-name=mrmc_experiment_prototype
#SBATCH --output=result-%A_%a.out
#SBATCH --error=result-%A_%a.err
#SBATCH --time=0-00:07
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=defq

export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2

source $1/../.env/bin/activate

srun -N 1 -n 1 --exclusive python $1/run_experiment.py --n_trials 30 --n_procs 31 --max_trials 320
