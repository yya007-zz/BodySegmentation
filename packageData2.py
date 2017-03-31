#--------------
# Created by Yuang Yao
# problem need to solved:
# 1 imd resize in float
# 2 registration
# should package it here
#--------------
import numpy as np
import scipy.io as sio
from PIL import Image
from scipy.misc import imresize
from scipy.misc import imsave

def load_image( infilename ) :
    img = Image.open( infilename )
    img = img.resize((512, 512), Image.NEAREST)
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data
    
    
    
dataNum=75
check=np.zeros([512])

images=[]
labels=[]
singleImage=True

def prepareY(y,number_of_classes):
    yf=y.flatten()
    print yf.shape
    print y.shape
    res=np.zeros([yf.shape[0],number_of_classes])
    for i in range(number_of_classes):
        res[yf[:]==i,i]=1
    return res.reshape([y.shape[0],y.shape[1],y.shape[2],number_of_classes])

def prepareX(rgb):
    VGG_MEAN = [103.939, 116.779, 123.68]
    res=np.zeros(rgb.shape)
    res[:,:,:,2]= rgb[:,:,:,0]-VGG_MEAN[2]
    res[:,:,:,1]= rgb[:,:,:,1]-VGG_MEAN[1]
    res[:,:,:,0]= rgb[:,:,:,2]-VGG_MEAN[0]    
    return res 
     
def grey2rmb(grey):
    res=np.zeros([grey.shape[0],grey.shape[1],3])
    res[:,:,0]=grey
    res[:,:,1]=grey
    res[:,:,2]=grey
    return res
    
    
def saveorgan(label,ind):
    yf=label.flatten()
    res=np.zeros([label.shape[0]*label.shape[1]])
    res[yf[:]==ind]=2
    res=res.reshape([label.shape[0],label.shape[1]])
    return imsave('organ.png',res)
    
    
#heart is 7
for i in range(dataNum):
    if i%10 == 0:
        print i
    image=sio.loadmat('./trainandtest/'+str(i+1)+'_img.mat')
    image=image['intensity']
    label=sio.loadmat('./trainandtest/'+str(i+1)+'_seg.mat')
    label=label['label']
    #imsave('bresize.png',image[:,243,:])
    if singleImage:
        count=np.zeros([image.shape[2]])
        for k in range(image.shape[2]):
            count[k]=np.sum(label[:,:,k]==7)
        select=np.argmax(count==np.max(count))
        #print select,np.max(count)
        
        image=grey2rmb(image[:,:,select])
        label=label[:,:,select]
        #imsave(str(i)+'organ.png',image)
      
    image=np.asarray(image).reshape([1,512,512,3])
    label=np.asarray(label).reshape([1,512,512])
    
    if i == 0:
        images=image
        labels=label
    else:
        images=np.concatenate((images,image), axis=0)
        labels=np.concatenate((labels,label), axis=0)
    
     
print images.shape,labels.shape,np.unique(labels)
print np.sum(images[0]-images[1])
np.save('../bigfile/img_3.npy',images)
np.save('../bigfile/seg_3.npy',labels)

