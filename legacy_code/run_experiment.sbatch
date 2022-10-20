#! /bin/bash

#SBATCH --job-name=alpha-corrections
#SBATCH --output=result-%A_%a.out
#SBATCH --error=result-%A_%a.err
#SBATCH --time=0-00:25
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=defq

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

source /mnt/nfs/home/jasonvallada/MRMC/.env/bin/activate

python /mnt/nfs/home/jasonvallada/MRMC/dice_path_test.py $1 $2
