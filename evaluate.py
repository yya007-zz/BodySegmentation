import tensorflow as tf
import FCN1
from PIL import Image
import numpy as np
import tensorflow as tf
import time
import sys
from scipy.misc import imsave
from test_bd import *
def load_image( infilename ) :
    img = Image.open( infilename )
    img = img.resize((512, 512), Image.NEAREST)      # use nearest neighbour
    #img = img.resize((width, height), Image.BILINEAR)     # linear interpolation in a 2x2 environment
    #img = img.resize((width, height), Image.BICUBIC)      # cubic spline interpolation in a 4x4 environment
    #img = img.resize((width, height), Image.ANTIALIAS)    # best down-sizing filter
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )

def next_batch(pos,size,data):
  if pos+size<data.shape[0]:
    return pos+size,data[pos:pos+size]
  else:
    return pos+size-data.shape[0],np.concatenate((data[pos:],data[0:pos+size-data.shape[0]]),axis=0)
    
def prepareY(y,number_of_classes):
    yf=y.flatten()
    #print yf.shape
    #print y.shape
    res=np.zeros([yf.shape[0],number_of_classes])
    for i in range(number_of_classes):
        res[yf[:]==i,i]=1
    return res.reshape([y.shape[0],y.shape[1],y.shape[2],number_of_classes])
    
    
#0-255 2d gray image
def prepareX(gray):
    #gray=gray.astype(int)
    #gray=gray.astype(float)
    VGG_MEAN = [103.939, 116.779, 123.68]
    res=np.zeros([gray.shape[0],512,512,3])
    res[:,:,:,2]= gray-VGG_MEAN[2]
    res[:,:,:,1]= gray-VGG_MEAN[1]
    res[:,:,:,0]= gray-VGG_MEAN[0]    
    return res 



def testall(sess,resdir='./',number_of_classes=19,objectNum=25,saveres=False):
    print ("start testing")
    
    if saveres:
        if not os.path.exists(resdir):
            os.makedirs(resdir)
    
    mydataFetch=dataFetch(4) 
    for objectInd in range(objectNum):
        label3D=np.zeros([512,512,512])
        for sliceInd in range(512):
            label3D[:,:,sliceInd]=mydataFetch.getImage(objectInd,2,sliceInd,'test','seg')
        predict3D=np.zeros([512,512,512,3])

        for viewInd in range(3):
        
            selectorder=np.arange(512)
            selectorder=selectorder+objectInd*viewNum*512+viewInd*512
            pos=0
            for sliceInd in range(512/size):
                startpos=pos
                pos,sample=next_batch(pos,size,selectorder)
                imgs=mydataFetch.getdata(sample,'test','img')
                segs=mydataFetch.getdata(sample,'test','seg')
                imgs=prepareX(imgs)
                segs=prepareY(segs,number_of_classes)
                if viewInd==0:
                    slicepre=sess.run(result,feed_dict={x: imgs, y_: segs, keep_prob: 1.0})
                    predict3D[startpos:startpos+size,:,:,0]=slicepre
                    sliceseg=label3D[startpos:startpos+size,:,:]
                if viewInd==1:
                    slicepre=sess.run(result,feed_dict={x: imgs, y_: segs, keep_prob: 1.0}).transpose(1,0,2)
                    predict3D[:,startpos:startpos+size,:,1]=slicepre
                    sliceseg=label3D[:,startpos:startpos+size,:]
                if viewInd==2:
                    slicepre=sess.run(result,feed_dict={x: imgs, y_: segs, keep_prob: 1.0}).transpose(1,2,0)
                    predict3D[:,:,startpos:startpos+size,2]=slicepre
                    sliceseg=label3D[:,:,startpos:startpos+size]
                print np.mean(sliceseg==slicepre)
        label3D=label3D.flatten()
        predict3DReal=np.zeros([512*512*512])
        predict3D=predict3D.reshape([512*512*512,3])
        
        accuracy=np.mean((predict3D[:,0]==label3D))
        accuracy1=np.mean((predict3D[:,1]==label3D))
        accuracy2=np.mean((predict3D[:,2]==label3D))
        print "object-%d view 0 accuracy: %.4f,view 1 accuracy: %.4f,view 2 accuracy: %.4f"%(objectInd,accuracy,accuracy1,accuracy2)
        
        
        predict3DReal=predict3D[:,2]
        needchange=(predict3D[:,0]==predict3D[:,1])
        predict3DReal[needchange]=predict3D[needchange,0]
           
        accuracy1=np.mean((predict3DReal==label3D))
        
        predict3DReal=predict3DReal[label3D!=0]
        label3D=label3D[label3D!=0]
        accuracy2=np.mean((predict3DReal==label3D))
        print "object-%d total accuracy: %.4f,only with label:%.4f"%(objectInd,accuracy1,accuracy2)
        if saveres:
            np.save(resdir+'%d.npy'%(objectInd),predict3DReal.reshape([512,512,512]))

    
    
epoch=1
randomstate='norandom'
quicktest='noquicktest'
#load network_________________________
number_of_classes=19
size=16
speed=1e-5

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None,512,512,3])
y_ = tf.placeholder(tf.float32, shape=[None,512,512,number_of_classes])
keep_prob = tf.placeholder(tf.float32)
y_conv=FCN1.FCN(x,keep_prob,number_of_classes=number_of_classes)
tf.add_to_collection("y_conv", y_conv)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(speed).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,3), tf.argmax(y_,3))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
result =tf.argmax(y_conv,3)
sess.run(tf.global_variables_initializer())


print ('start loading model_%d_%s'%(epoch,randomstate)) 
modelname=('model_%d_%s'%(epoch,randomstate))
modeldir=('../network/%s/%s'%(modelname,modelname))
new_saver = tf.train.import_meta_graph(modeldir+'.meta')
new_saver.restore(sess,modeldir)



#testing---------------------------
objectNum=25
viewNum=3
mydataFetch=dataFetch(4)



'''
print ("start testing")  
imgs=np.load('../bigfile/testimgs.npy')
segs=np.load('../bigfile/testsegs.npy')
print sess.run(accuracy,feed_dict={x: imgs, y_: segs, keep_prob: 1.0})
 
  
if True:
    selectorder=np.arange(0,objectNum*viewNum*512,viewNum*512)
    selectorder=selectorder+2*512+256
    pos=0
    for k in range(len(selectorder)/size):
        pos,sample=next_batch(pos,size,selectorder)       
        imgs=mydataFetch.getdata(sample,'test','img')
        segs=mydataFetch.getdata(sample,'test','seg')
        imgs=prepareX(imgs)
        segs=prepareY(segs,number_of_classes)
        print "test"+str(k)
        print sess.run(accuracy,feed_dict={x: imgs, y_: segs, keep_prob: 1.0})
'''

     
resdir='../res/%s_%s_%s/'%(randomstate,epoch,quicktest)
testall(sess,resdir=resdir,number_of_classes=number_of_classes)

