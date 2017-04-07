import numpy as np
from scipy import misc
import scipy.io as sio


def dataFetch(class):
    
    def __init___()
        pass

    def int2string(number,length=4):
        if length==0:
            return ""   
        return int2string(number/10,length-1)+str(number%10)
        


    '''
    def getImage(objectInd,viewInd,sliceInd,dataset,subset):
        imgdir='../Data2D/%s/%s/sub_%s/view%d/slice_%s.png'%(dataset,subset,int2string(objectInd+1),viewInd+1,int2string(sliceInd+1))
        image = misc.imread(imgdir)
        img=np.asarray(image)
        if subset=='img':
            img=img.astype(float)*(1.0/65535.0*255.0)
        return img
    '''    
    def getImage(objectInd,viewInd,sliceInd,dataset,subset):
        imgdir='../Data3D/%s/%s/sub_%s.mat'%(dataset,subset,int2string(objectInd+1))
        img=sio.loadmat(imgdir)
        img=img[subset+'_3D']
        if viewInd==0:
            image=img[sliceInd,:,:]
        if viewInd==1:
            image=img[:,sliceInd,:]
        if viewInd==2:
            image=img[:,:,sliceInd]
        if subset=='img':
            image=image.astype(float)*(1.0/65535.0*255.0)
        return image    
       

    def getdata(selectorder,dataset,subset):
        viewNum=3
        resultimg=np.zeros([len(selectorder),512,512])
        for i in range(len(selectorder)):
            objectInd=selectorder[i]/(viewNum*512)
            viewInd=(selectorder[i]-objectInd*viewNum*512)/512
            sliceInd=selectorder[i]%512
            resultimg[i,:,:]=getImage(objectInd,viewInd,sliceInd,dataset,subset)
        return resultimg
    
    
    
    
'''
dataset='train'
subset='seg'    
objectNum=75
viewNum=3
selectorder=np.arange(objectNum*viewNum*512)
#selectorder=np.random.shuffle(selectorder)
seg=getdata(selectorder[512:512+200],dataset,subset)
objectInd=0
imgdir='../Data3D/%s/%s/sub_%s.mat'%(dataset,subset,int2string(objectInd+1))
img=sio.loadmat(imgdir)
img=img[subset+'_3D']
image=img[:,0:200,:]
if subset=='img':
    image=image.astype(float)*(1.0/65535.0*255.0)
print np.unique(img)
print np.sum(image[100]-seg[:,100,:])
print np.sum(image-seg.transpose(1,0,2))




objectInd=1
accuracy1=0.00001
accuracy2=0.0002
print "object-%d accuracy no label: %.4f,with label:%.4f"%(objectInd,accuracy1,accuracy2)

objectNum=75
viewNum=3
selectorder=np.arange(objectNum*viewNum*512)
np.random.shuffle(selectorder)
print selectorder
'''







