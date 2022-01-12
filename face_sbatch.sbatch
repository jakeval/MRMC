#! /bin/bash

#SBATCH --job-name=face_graph
#SBATCH --output=result-%A_%a.out
#SBATCH --error=result-%A_%a.err
#SBATCH --time=0-02:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2000
#SBATCH --partition=defq

export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2

module load shared python/3.8.11

source /home/jasonvallada/MRMC/.env/bin/activate

python /home/jasonvallada/MRMC/graph_series.py $1 $2 $3 $4 $5
