#! /bin/bash
#SBATCH --account=p_masi_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --time=5:00:00
#SBATCH --output=/scratch/yaoy4/log/resave.log


setpkgs -a tensorflow_0.12
python  /scratch/yaoy4/BodySegmentation/testresave.py ../res/norandomrun_0/
python  /scratch/yaoy4/BodySegmentation/testresave.py ../res/norandomrun_5/
python  /scratch/yaoy4/BodySegmentation/testresave.py ../res/norandomrun_1/
python  /scratch/yaoy4/BodySegmentation/testresave.py ../res/norandomrun_2/
python  /scratch/yaoy4/BodySegmentation/testresave.py ../res/norandomrun_3/
python  /scratch/yaoy4/BodySegmentation/testresave.py ../res/norandomrun_4/
