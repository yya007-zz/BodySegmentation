#! /bin/bash
#SBATCH --account=p_masi_gpu
#SBATCH --partition=maxwell
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/yaoy4/log/quicktest.log
 
python test.py 
python test.py 
python test.py 
python test.py 




