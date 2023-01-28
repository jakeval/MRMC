#! /bin/bash

rm result-*     # remove previous stdout and stderr logs

# Arg $1: The config file to run. Path can be relative.
# Arg $2: number of processes
# Arg $3: max_runs. Optional.
sbatch --ntasks $2 run.sbatch $2 /mnt/nfs/scratch1/$USER $1 $2
