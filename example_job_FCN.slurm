#! /bin/bash
#SBATCH --account=p_masi_gpu
#SBATCH --partition=maxwell
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=5G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/huoy1/projects/DeepLearning/FCN/log/gpu-job.log

setpkgs -a tensorflow_0.12
source activate FCN
cd /scratch/huoy1/projects/DeepLearning/FCN/BodySegmentation
python test_yh.py
