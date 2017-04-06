#! /bin/bash
#SBATCH --account=p_masi_gpu
#SBATCH --partition=maxwell
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/yaoy4/log/test-1echo-norandom.log

setpkgs -a tensorflow_0.12
source activate FCN
cd /scratch/yaoy4/BodySegmentation 
python test_tf.py norandom 1 noquicktest
