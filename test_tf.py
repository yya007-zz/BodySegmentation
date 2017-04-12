import tensorflow as tf
import FCN1
from PIL import Image
import numpy as np
import tensorflow as tf
import time
import sys
import os
from scipy.misc import imsave
from sklearn.model_selection import train_test_split
from test_bd import *
from evaluate import *
# python test_tf [random?] [echo] [quicktest]


objectNum=75
viewNum=3


print "----------------start setting environment"
selectorder=np.arange(objectNum*viewNum*512)


randomstate="norandom"
if sys.argv[1]=='random':
    randomstate="random"
    np.random.shuffle(selectorder)

size=16
epoch=int(sys.argv[2])
iterations=epoch*len(selectorder)/size

quicktest=False
if sys.argv[3]=='quicktest':
    quicktest=True
    selectorder=np.arange(0,objectNum*viewNum*512,viewNum*512)
    selectorder=selectorder+2*512+256
    iterations=epoch    
gap=int(iterations/100)
if gap<3:
    gap=3

mydataFetch=dataFetch(25)

print "----------------start building network"
number_of_classes=19
speed=1e-5
#Network structure--------------------------                                 
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None,512,512,3])
y_ = tf.placeholder(tf.float32, shape=[None,512,512,number_of_classes])
keep_prob = tf.placeholder(tf.float32)
y_conv=FCN1.FCN(x,keep_prob,number_of_classes=number_of_classes)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(speed).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,3), tf.argmax(y_,3))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
result =tf.argmax(y_conv,3)
saver0 = tf.train.Saver()
sess.run(tf.global_variables_initializer())

#training--------------------------
print "----------------start training"
pos=0
t0 = time()
for i in range(iterations):
    if sys.argv[3]=='quicktest':
        imgs=np.load('../bigfile/testimgs.npy')
        segs=np.load('../bigfile/testsegs.npy')
    else:
        pos,sample=next_batch(pos,size,selectorder)    
        imgs=mydataFetch.getdata(sample,'train','img')
        segs=mydataFetch.getdata(sample,'train','seg')
        imgs=prepareX(imgs)
        segs=prepareY(segs,number_of_classes)
    #print "step: ",i
    
    if i==0:
        print "traindata: %d randomstate: %s, echo,iterations: %d,%d, gap: %d "%(len(selectorder),randomstate,epoch,iterations,gap)
    if i%gap == 0 or i==iterations-1:
        cp,ce=sess.run([correct_prediction,cross_entropy],feed_dict={x: imgs, y_: segs,keep_prob: 1.0})
        ac=np.mean(cp)
        ac2=np.mean(cp[1:])
        print("step %d, training accuracy %d, only label: %d, loss %g, time %d"%(i, ac,ac2,ce,time()-t0))
        t0 = time()
    train_step.run(feed_dict={x: imgs, y_: segs, keep_prob: 0.5})
del imgs,segs,mydataFetch


#testing---------------------------


print ("----------------start saving model")
objectNum=25
viewNum=3 

'''
if sys.argv[3]=='quicktest':
    selectorder=np.arange(0,objectNum*viewNum*512,viewNum*512)
    selectorder=selectorder+2*512+256
    pos=0
    for k in range(len(selectorder)/size):
        pos,sample=next_batch(pos,size,selectorder)       
        imgs=mydataFetch.getdata(sample,'test','img')
        segs=mydataFetch.getdata(sample,'test','seg')
        imgs=prepareX(imgs)
        segs=prepareY(segs,number_of_classes)
        print accuracy.eval(feed_dict={x: imgs, y_: segs, keep_prob: 1.0})
'''  



modelname=('model_%d_%s_%s'%(epoch,randomstate,sys.argv[3]))
modeldir=('../network/'+modelname)  
if not os.path.exists(modeldir):
    os.makedirs(modeldir)
modeldir=(modeldir+'/'+modelname)  
print 'save model to: %s'%(modeldir)
saver0.save(sess, modeldir)
saver0.export_meta_graph(modeldir+'.meta')

    

resdir='../res/%s_%s_%s/'%(sys.argv[1],sys.argv[2],sys.argv[3])
testall(sess,result,x,y_,keep_prob,quicktest=quicktest,resdir=resdir,number_of_classes=number_of_classes,saveres=True)




