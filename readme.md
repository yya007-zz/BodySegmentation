# Body Segmentation Using Fully Convolutional Networks 

## Introduction
This project is to use fully convolutional networks to do object segmentation from images. This project is primarily used for organ segmentations from CT images, but it could also transfer to do general object segnmentations from any type of images based on given labeled images.</br>


This project is written by Python 2.7 and tensorflow 0.12.</br>

## Enviroment Setup
This project uses the weights used in VGG16. The weights and pretrained parameters are stored in a numpy file.The .npy file for [VGG16] to be downloaded before using this needwork. You can find the file here: ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy




## For Vanderbilt User


### For ACCRE User


#### Step 1 ssh to accre

ssh VUnetid@login.accre.vanderbilt.edu

#### Step 2 load tensorflow 0.12
setpkgs -a tensorflow_0.12

#### Step 3 setup your anaconda envrioment
First setup:
setpkgs -a anaconda2
conda create --name FCN python=2.7
source activate FCN
pip install keras
pip install protobuf
pip install matplotlib
pip install pillow

If already setup environment:
source activate FCN


#### step 4(Optional) download from github
setpkgs -a git


#### Step 5(Optinonal) copy from other machine


#### Step 5(Optinonal) visualize testing 
request GPU node
salloc --account=p_masi_gpu  --partition=maxwell --ntasks=4 --nodes=1 --gres=gpu:2 --time=5:00:00 --mem=40G
cd /scatch/...
python run.py



