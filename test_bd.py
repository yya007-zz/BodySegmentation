import numpy as np
from scipy import misc
import scipy.io as sio
from time import time

class dataFetch(object):
    
    def __init__(self):
        self.tempStore=[]
        self.tempStoreName=["",""]
        self.nextsave=0
        self.savelen=2
        self.cursave=0
        pass

    def int2string(self,number,length=4):
        if length==0:
            return ""   
        return self.int2string(number/10,length-1)+str(number%10)

    '''
    def getImage(objectInd,viewInd,sliceInd,dataset,subset):
        imgdir='../Data2D/%s/%s/sub_%s/view%d/slice_%s.png'%(dataset,subset,int2string(objectInd+1),viewInd+1,int2string(sliceInd+1))
        image = misc.imread(imgdir)
        img=np.asarray(image)
        if subset=='img':
            img=img.astype(float)*(1.0/65535.0*255.0)
        return img
    '''    
    
    def getImage(self,objectInd,viewInd,sliceInd,dataset,subset):
        imgdir='../Data3D/%s/%s/sub_%s.mat'%(dataset,subset,self.int2string(objectInd+1))
        flag=True
        for i in range(len(self.tempStoreName)):
            if imgdir==self.tempStoreName[i]:
                img=self.tempStore[i]
                flag=False
        if flag:
            img=sio.loadmat(imgdir)
            img=img[subset+'_3D']
            
            if self.cursave<self.savelen:
                self.tempStore.append(img)
                self.cursave=self.cursave+1
            else:
                self.tempStore[self.nextsave]=img  
            self.tempStoreName[self.nextsave]==imgdir
              
            self.nextsave=(self.nextsave+1)%self.savelen
            
        if viewInd==0:
            image=img[sliceInd,:,:]
        if viewInd==1:
            image=img[:,sliceInd,:]
        if viewInd==2:
            image=img[:,:,sliceInd]
        if subset=='img':
            image=image.astype(float)*(1.0/65535.0*255.0)
        return image    
       

    def getdata(self,selectorder,dataset,subset):
        viewNum=3
        resultimg=np.zeros([len(selectorder),512,512])
        for i in range(len(selectorder)):
            objectInd=selectorder[i]/(viewNum*512)
            viewInd=(selectorder[i]-objectInd*viewNum*512)/512
            sliceInd=selectorder[i]%512
            resultimg[i,:,:]=self.getImage(objectInd,viewInd,sliceInd,dataset,subset)
        return resultimg
    
    
    
''' 
mydataFetch=dataFetch()
dataset='train'
subset='seg'    
objectNum=75
viewNum=3
selectorder=np.arange(objectNum*viewNum*512)
#selectorder=np.random.shuffle(selectorder)
seg=mydataFetch.getdata(selectorder[512:512+200],dataset,subset)
objectInd=0
imgdir='../Data3D/%s/%s/sub_%s.mat'%(dataset,subset,mydataFetch.int2string(objectInd+1))
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







