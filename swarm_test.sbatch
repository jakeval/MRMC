#! /bin/bash

#SBATCH --job-name=jasonvallada_test
#SBATCH --output=result-%A_%a.out
#SBATCH --error=result-%A_%a.err
#SBATCH --time=0-00:40
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=30
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500
#SBATCH --partition=defq
#SBATCH --mail-user=$USER@umass.edu

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

module load shared python/3.8.11

source /home/jasonvallada/MRMC/.env/bin/activate

python /home/jasonvallada/MRMC/alpha_test.py $1
