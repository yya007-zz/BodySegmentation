#! /bin/bash
#SBATCH --account=p_masi_gpu
#SBATCH --partition=maxwell
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=40G
#SBATCH --time=120:00:00
#SBATCH --output=/scratch/yaoy4/log/test-norandom-list13.log
setpkgs -a tensorflow_0.12
python  /scratch/yaoy4/BodySegmentation/run.py train norandom 13
python  /scratch/yaoy4/BodySegmentation/run.py train norandom 13
python  /scratch/yaoy4/BodySegmentation/run.py train norandom 13
python  /scratch/yaoy4/BodySegmentation/run.py train norandom 13
python  /scratch/yaoy4/BodySegmentation/run.py train norandom 13
python  /scratch/yaoy4/BodySegmentation/run.py train norandom 13
python  /scratch/yaoy4/BodySegmentation/run.py train norandom 13
