#!/bin/bash
#SBATCH --mail-user=yaoy4@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=2G
#SBATCH --array=0-99
#SBATCH --output=/scratch/yaoy4/log/Array_job_slurm_%A_%a.out

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

setpkgs -a python2.7.8
python < package_single.py ${SLURM_ARRAY_TASK_ID}
