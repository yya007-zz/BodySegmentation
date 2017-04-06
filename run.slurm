#! /bin/bash
#SBATCH --account=p_masi_gpu
#SBATCH --partition=maxwell
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=20G
#SBATCH --time=120:00:00
#SBATCH --output=/scratch/yaoy4/log/test-10echo-random.log

setpkgs -a tensorflow_0.12
source activate FCN
cd /scratch/yaoy4/BodySegmentation 
python test_tf.py random 10 noquicktest
