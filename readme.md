# Body Segmentation Using Fully Convolutional Networks 

##
Please star our projects if you think it is useful!

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
```
git https://github.com/yya007/BodySegmentation.git
cd BodySegmentation
python setup.py
```
Then, move the VGG16.npy to the bigfile folder</br>
The network folder is for storing trained network with different settings and epoch number.</br>
The res folder is for storing results by evaluating the performance of network</br>
The Data3D folder is for storing train and test data. In test and train folder, all input images are divided into two category: seg and img. Each subject should in .mat form and named as sub_[four digit index with 1 indexing].mat with dimension 512,512,512</br>
For exmaple, the label of first train object should in `/Data3D/test/Seg/sub_0001.mat` and in this mat file ['seg'] has dimension of 512,512,512.</br>
#### Run the code
```
python run.py train [random/norandom] [number_of_classes]
```
will keep train from last trained networks with same parameters
#### Evaluate results
##### Evaluate the performance of network
```
python run.py evaluate [random/norandom] [number_of_classes]
```
will evaluate the last unevaluate network with same parameters
##### Plot the box plot
```
python plot.py [corresponding result address]
```
For example, if you want to plot the box plot of `randomrun_0` (the model after first epoch with random shuffle) under `res` folder</br>
```
python plot.py ../res/randomrun_0
```
Then you will get the box plot of dice similarity of each organs between ground truth and predication made by model.

## Related Publication
* Yuang Yao, Yuankai Huo, Zhoubing Xu, Bennett Landman. "Fully Convolutional Networks Enables the Simultaneous Abdominal Organs Segmentation with Muscle, Fat and Bones from Clinical Acquired CT Images" (Pending)
* Meg F. Bobo, Shunxing Bao, Yuankai Huo, Yuang Yao, Jack Virostko, Melissa A. Hilmes, Andrew J. Plassard, Ilwoo Lyu, Albert Assad, Richard G. Abramson, Bennett A. Landman. "Fully Convolutional Neural Networks Improve Abdominal Organ Segmentation." In SPIE Medical Imaging. International Society for Optics and Photonics, 2018.
* Shunxing Bao, Yuankai Huo, Prasanna Parvathaneni, Andrew J. Plassard, Camilo Bermudez, Yuang Yao, Ilwoo Lyu, Aniruddha Gokhale and Bennett A. Landman. "A Data Colocation Grid Framework for Big Data Medical Image Processing - Backend Design." In SPIE Medical Imaging, International Society for Optics and Photonics, 2018.  
* Yuankai Huo, Zhoubing Xu, Shunxing Bao, Camilo Bermudez, Andrew J. Plassard, Jiaqi Liu, Yuang Yao, Albert Assad, Richard G. Abramson, and Bennett A. Landman. " Splenomegaly segmentation using global convolutional kernels and conditional generative adversarial networks." In SPIE Medical Imaging, International Society for Optics and Photonics, 2018.

## PS: For Vanderbilt User
### For ACCRE User
#### Step 1 ssh to accre
```
ssh VUnetid@login.accre.vanderbilt.edu
```
#### Step 2 load tensorflow 0.12
```
setpkgs -a tensorflow_0.12
```
#### Step 3 setup your anaconda envrioment
First setup:</br>
```
setpkgs -a anaconda2
conda create --name FCN python=2.7
source activate FCN
pip install keras
pip install protobuf
pip install matplotlib
pip install pillow
```
If already setup environment:</br>
```
source activate FCN
```
#### Step 4 (Optional) download from github
```
setpkgs -a git
git https://github.com/yya007/BodySegmentation.git
```
#### Step 5 (Optinonal) copy from other machine
```
scp -r [your machine address]:[folder in local machine]   [target folder on ACCRE]
```

#### Step 6 (Optinonal) visualize testing 
request GPU node:</br>
```
salloc --account=p_masi_gpu  --partition=maxwell --ntasks=4 --nodes=1 --gres=gpu:2 --time=5:00:00 --mem=40G
cd /scatch/...
python run.py
```
#### Step 7 (Optinonal) submit slurm jobs 
sample slurm files are in ACCRE folder

## Contact
Email yuang.yao@vanderbilt.edu


