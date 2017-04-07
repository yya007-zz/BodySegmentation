#! /bin/bash
#SBATCH --account=p_masi_gpu
#SBATCH --partition=maxwell
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=20G
#SBATCH --time=120:00:00
#SBATCH --output=/scratch/yaoy4/log/test-10echo-norandom.log

setpkgs -a tensorflow_0.12
source activate FCN
cd /scratch/yaoy4/BodySegmentation 
python test_tf.py norandom 10 noquicktest
