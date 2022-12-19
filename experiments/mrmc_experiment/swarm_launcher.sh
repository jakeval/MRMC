#! /bin/bash

rm result-*     # For convenience, remove previous stdout and stderr logs

# Arg $1: number of processes
# Arg $2: max_runs. Optional.
sbatch --ntasks $1 run.sbatch $1 /mnt/nfs/scratch1/$USER $2
