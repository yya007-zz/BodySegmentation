#! /bin/bash
#SBATCH --account=p_masi_gpu
#SBATCH --partition=maxwell
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --array=0-2
#SBATCH --output=/scratch/yaoy4/log/python_array_job_slurm_%A_%a.out
setpkgs -a tensorflow_0.12
source activate FCN

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

cd /scratch/yaoy4/BodySegmentation
python package_data.py ${SLURM_ARRAY_TASK_ID}
