import numpy as np
from scipy import misc


def int2string(number,length=4):
    if length==0:
        return ""   
    return int2string(number/10,length-1)+str(number%10)
    



def getImage(objectInd,viewInd,sliceInd,dataset,subset):
    imgdir='../Data2D/%s/%s/sub_%s/view%d/slice_%s.png'%(dataset,subset,int2string(objectInd+1),viewInd+1,int2string(sliceInd+1))
    image = misc.imread(imgdir)
    img=np.asarray(image)
    if subset=='img':
        img=img.astype(float)*(1.0/65535.0*255.0)
    return img
    
    
   

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
subset='img'    
objectNum=75
viewNum=3
selectorder=np.arange(objectNum*viewNum*512)
#selectorder=np.random.shuffle(selectorder)
seg=getdata(selectorder[1:4],'test','img')

    
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
