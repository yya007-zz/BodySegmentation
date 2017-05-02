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
    
    
def transY(gray):
    return 0

for i in range(dataNum):
    if i%10 == 0:
        print i
    image=sio.loadmat('./trainandtest/'+str(i+1)+'_img.mat')
    image=image['intensity']
    label=sio.loadmat('./trainandtest/'+str(i+1)+'_seg.mat')
    label=label['label']
    #imsave('bresize.png',image[:,243,:])
    if singleImage:
        image=grey2rmb(image[:,243,:])
        label=label[:,243,:]
        
    
    
    # Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic')
    image= imresize(image,[512,512], interp='nearest')
    #print '1',len(np.unique(label)),np.unique(label)
    #imsave('bresize.png',label),
    label= imresize(label,[512,512], interp='nearest')
    #print '2',len(np.unique(label)),np.unique(label)
    #imsave('aresize.png',label)
    #print image,np.asarray(image).shape
    #imsave('aresize.png',label)
    
    image=np.asarray(image).reshape([1,512,512,3])
    label=np.asarray(label).reshape([1,512,512])
    
    if i == 0:
        images=image
        labels=label
    else:
        images=np.concatenate((images,image), axis=0)
        labels=np.concatenate((labels,label), axis=0)
    
    for k in range(512):
        check[k]+=len(np.unique(label[:,k,:].flatten()))
    print len(np.unique(label[:,:,:].flatten())),len(np.unique(label[:,200,:].flatten()))
   
    
print np.max(check)*1.0/75,np.argmax(check==np.max(check))
   
for i in range(dataNum):
    if i%10 == 0:
        print i
    image=sio.loadmat('./trainandtest/'+str(i+1)+'_img.mat')
    image=image['intensity']
    label=sio.loadmat('./trainandtest/'+str(i+1)+'_seg.mat')
    label=label['label']

    for k in range(512):
        check[k]+=len(np.unique(label[:,k,:].flatten()))
    print len(np.unique(label[:,:,:].flatten())),len(np.unique(label[:,200,:].flatten()))
   
    
print np.max(check)*1.0/75,np.argmax(check==np.max(check))


print images.shape,labels.shape,np.unique(labels)
print np.sum(images[0]-images[1])
np.save('../bigfile/img_3.npy',images)
np.save('../bigfile/seg_3.npy',labels)
