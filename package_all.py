#--------------
# Created by Yuang Yao
# problem need to solved:
# 1 imd resize in float
# 2 registration
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
    
    
def transY(gray):
    return 0

for i in range(dataNum):
    assert i==0
    if i%10 == 0:
        print i
    image=sio.loadmat('./train/'+str(i+1)+'_img.mat')
    oriimage=image['intensity']
    label=sio.loadmat('./train/'+str(i+1)+'_seg.mat')
    orilabel=label['label']
    for j in range(3):
        current_shape=(orilabel.shape[j])
        for k in range(current_shape):
            needResize=True
            if j==0:
                image=oriimage[k,:,:]
                label=orilabel[k,:,:]
            elif j==1:
                image=oriimage[:,k,:]
                label=orilabel[:,k,:]
            else:
                image=oriimage[:,:,k]
                label=orilabel[:,:,k]
                needResize=False
                
            if needResize:  
                # Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic')
                image= imresize(image,[512,512], interp='nearest')
                label= imresize(label,[512,512], interp='nearest')
    
            image=np.asarray(image).reshape([1,512,512])
            label=np.asarray(label).reshape([1,512,512])
            
            if  j==0 and k==0:
                images=image
                labels=label
            else:
                images=np.concatenate((images,image), axis=0)
                labels=np.concatenate((labels,label), axis=0)
    
            if k%100==0:
                print 'i',i,'j',j,'k',k,images.shape
    
    print images.shape,labels.shape,np.unique(labels)
    print np.sum(images[0]-images[1])
    np.save('../bigfile/train/img'+str(i+1)+'.npy',images)
    np.save('../bigfile/train/seg'+str(i+1)+'.npy',labels)

dataNum=25
for i in range(dataNum):
    if i%10 == 0:
        print i
    image=sio.loadmat('./train/test/'+str(i+1)+'_img.mat')
    oriimage=image['intensity']
    label=sio.loadmat('./train/test/'+str(i+1)+'_seg.mat')
    orilabel=label['label']
    for j in range(3):
        current_shape=(orilabel.shape[j])
        for k in range(current_shape):
            needResize=True
            if j==0:
                image=oriimage[k,:,:]
                label=orilabel[k,:,:]
            elif j==1:
                image=oriimage[:,k,:]
                label=orilabel[:,k,:]
            else:
                image=oriimage[:,:,k]
                label=orilabel[:,:,k]
                needResize=False
                
            if needResize:  
                # Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic')
                image= imresize(image,[512,512], interp='nearest')
                label= imresize(label,[512,512], interp='nearest')
    
            image=np.asarray(image).reshape([1,512,512])
            label=np.asarray(label).reshape([1,512,512])
    
            if j==0 and k==0:
                images=image
                labels=label
            else:
                images=np.concatenate((images,image), axis=0)
                labels=np.concatenate((labels,label), axis=0)
        
    print images.shape,labels.shape,np.unique(labels)
    print np.sum(images[0]-images[1])
    np.save('../bigfile/test/img'+str(i+1)+'.npy',images)
    np.save('../bigfile/test/seg'+str(i+1)+'.npy',labels)




