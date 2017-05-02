# Body Segmentation Using Fully Convolutional Networks 

## Introduction
This project is to use fully convolutional networks to do object segmentation from images. This project is primarily used for organ segmentations from CT images, but it could also transfer to do general object segnmentations from any type of images based on given labeled images.</br>


This project is written by Python 2.7 and tensorflow 0.12.</br>

## Instruction

### Enviroment Setup
#### Suport files
This project uses the weights used in VGG16. The weights and pretrained parameters are stored in a numpy file.The .npy file for [VGG16] to be downloaded before using this needwork. You can find the file here: ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy 
#### Required Packages
Install Python 2 / Install annaconda 2</br>
Install tensorflow 0.12</br>
#### Data preprocessing
git https://github.com/yya007/BodySegmentation.git</br>
cd BodySegmentation</br>
python setup.py</br>
Then, move the VGG16.npy to the bigfile folder</br>
The network folder is for storing trained network with different settings and epoch number.</br>
The res folder is for storing results by evaluating the performance of network</br>
The Data3D folder is for storing train and test data. In test and train folder, all input images are divided into two category: seg and img. Each subject should in .mat form and named as sub_[four digit index with 1 indexing].mat with dimension 512,512,512</br>
For exmaple, the label of first train object should in /Data3D/test/Seg/sub_0001.mat and in this mat file ['seg'] has dimension of 512,512,512.</br>
#### Run the code
python run.py train [random/norandom] [number_of_classes]</br>
will keep train from last trained networks with same parameters
#### Evaluate results
##### Evaluate the performance of network
python run.py evaluate [random/norandom] [number_of_classes]</br>
will evaluate the last unevaluate network with same parameters
##### Plot the box plot
'python plot.py [corresponding result address]'</br>
For example, if you want to plot the box plot of randomrun_0 under res folder</br>
python plot.py ../res/randomrun_0</br>

## For Vanderbilt User
### For ACCRE User
#### Step 1 ssh to accre
ssh VUnetid@login.accre.vanderbilt.edu</br>
#### Step 2 load tensorflow 0.12
setpkgs -a tensorflow_0.12</br>
#### Step 3 setup your anaconda envrioment
First setup:</br>
setpkgs -a anaconda2</br>
conda create --name FCN python=2.7</br>
source activate FCN</br>
pip install keras</br>
pip install protobuf</br>
pip install matplotlib</br>
pip install pillow</br>

If already setup environment:</br>
source activate FCN</br>
#### Step 4(Optional) download from github
setpkgs -a git</br>
git https://github.com/yya007/BodySegmentation.git
#### Step 5(Optinonal) copy from other machine
scp -r [your machine address]:[folder in local machine]   [target folder on ACCRE]</br>

#### Step 6(Optinonal) visualize testing 
request GPU node:</br>
salloc --account=p_masi_gpu  --partition=maxwell --ntasks=4 --nodes=1 --gres=gpu:2 --time=5:00:00 --mem=40G
cd /scatch/...
python run.py
#### Step 6(Optinonal) submit slurm jobs 
sample slurm jobs is in slurm folder
## For Future Developer 
Email yuang.yao@vanderbilt.edu


