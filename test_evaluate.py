import tensorflow as tf
import FCN1
from PIL import Image
import numpy as np
import tensorflow as tf
import time
import sys
from scipy.misc import imsave
from test_bd import * 
from evaluate import *  
    
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


print ('start loading model_%d_%s_%s'%(epoch,randomstate,quicktest)) 
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
testall(sess,result,x,y_,keep_prob,resdir=resdir,printstep=True,number_of_classes=number_of_classes,saveres=True)
