#! /bin/bash

#SBATCH --job-name=face_experiment
#SBATCH --output=result-%A_%a.out
#SBATCH --error=result-%A_%a.err
#SBATCH --time=0-00:07
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=defq

export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2

source ./../../.env/bin/activate

let n_procs=$1-1

if [ -z $4 ]
then
    max_runs=""
else
    max_runs="--max_runs ${4}"
fi

srun -N 1 -n 1 --exclusive python ./run_face_experiment.py \
--experiment \
--verbose \
--config $3 \
--n_procs $n_procs \
--slurm \
--distributed \
--scratch_dir $2 \
$max_runs
