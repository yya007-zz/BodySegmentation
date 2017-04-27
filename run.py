import tensorflow as tf
import numpy as np
import FCN1
from time import time
import os
import sys
from test_bd import dataFetch
from evaluate import testall,next_batch,prepareX,prepareY,randomshuffle



def savemodel(modelname,saver,sess):
    print ("----------------start saving model")
    modeldir=('../network/'+modelname)  
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    modeldir=(modeldir+'/'+modelname)  
    print 'save model to: %s'%(modeldir)
    saver.save(sess, modeldir)



def trainEpoch(evaluate=True,train=True,restore=True,save=True,rand=False):
    objectNum=75
    viewNum=3
    selectorder=np.arange(objectNum*viewNum*512)
    
    state="run"
    size=16
    quicktest=False
    storelength=4

    if state=='quicktest':
        quicktest=True
        selectorder=np.arange(0,objectNum*viewNum*512,viewNum*512)
        selectorder=selectorder+2*512+256
        storelength=4


        
    if rand:
        objectorder=np.arange(objectNum)
        objectorder=randomshuffle(objectorder)
        for i in range(objectNum):
            temporder=randomshuffle(np.arange(viewNum*512))
            selectorder[i*viewNum*512:(i+1)*viewNum*512]=temporder+objectorder[i]*viewNum*512
        assert len(np.unique(selectorder))==objectNum*viewNum*512
        
        
        
        state="random"+state
    else:
        state="norandom"+state

    iterationsOne=len(selectorder)/size
    gap=iterationsOne//10
    mydataFetch=dataFetch(storelength)
    if gap<1:
        gap=1
    number_of_classes=19
        
    
    
    #get newest trained network
    epochind=0
    modelname=('model_%s_%d'%(state,epochind))
    modelfolddir=('../network/'+modelname) 
    while os.path.exists(modelfolddir):
        epochind+=1
        modelname=('model_%s_%d'%(state,epochind))
        modelfolddir=('../network/'+modelname)
    epochind=epochind-1
    
    #Network structure-------------------------- 
    
    print "----------------start building network"                                
    x = tf.placeholder(tf.float32, shape=[None,512,512,3])
    y_ = tf.placeholder(tf.float32, shape=[None,512,512,number_of_classes])
    keep_prob = tf.placeholder(tf.float32)
    speed=tf.placeholder(tf.float32)
    
    
    y_conv=FCN1.FCN(x,keep_prob,number_of_classes=number_of_classes)
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    train_step = tf.train.AdamOptimizer(speed).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,3), tf.argmax(y_,3))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result =tf.argmax(y_conv,3)
    saver = tf.train.Saver()

        
    
    imgs=np.load('../bigfile/testimgs.npy')
    segs=np.load('../bigfile/testsegs.npy')
    print "traindata: %d state: %s,iterations%d, gap: %d "%(len(selectorder),state,iterationsOne,gap)
    
    
    
    with tf.Session() as sess:
        t0 = time()   
        sess.run(tf.global_variables_initializer())        
        if train: 
            print "----------------start training"
            if  epochind>=0:
                print ('start loading model_%s_%d'%(state,epochind)) 
                modelname=('model_%s_%d'%(state,epochind))           
                modeldir=('../network/%s/%s'%(modelname,modelname))
                saver.restore(sess,modeldir)
            
        
            epochind=epochind+1               
            modelname=('model_%s_%d'%(state,epochind))
            print "need new model",modelname
            pos=0
            
            
            trainspeed=1e-5
            if epochind>6:
                trainspeed=1e-7

            print "training speed is",trainspeed
            
            for iterind in range(iterationsOne):
                #if True:
                if not quicktest:
                    pos,sample=next_batch(pos,size,selectorder)    
                    imgs=mydataFetch.getdata(sample,'train','img')
                    segs=mydataFetch.getdata(sample,'train','seg')
                    imgs=prepareX(imgs)
                    segs=prepareY(segs,number_of_classes)
                train_step.run(feed_dict={x: imgs, y_: segs, keep_prob: 0.5,speed:trainspeed}) 
                if iterind%gap == 0 or iterind==iterationsOne-1:
                    (cp,ce)=(correct_prediction,cross_entropy).eval(feed_dict={x: imgs, y_: segs,keep_prob: 1.0,speed:trainspeed})
                    #ce=cross_entropy.eval(feed_dict={x: imgs, y_: segs,keep_prob: 1.0})
                    ac=np.mean(cp)
                    ac2=np.mean(cp[1:])
                    print("epoch: %d,step: %d, training accuracy %.4f, only label: %.4f, loss %g, time %d"%(epochind,iterind, ac,ac2,ce,time()-t0))
                    t0 = time()   
            if save:
                savemodel(modelname,saver,sess)
                print "successfully save model"
        else:
            print "----------------start without training"
            #skip evaluted network
            epochind=0
            resdir='../res/%s_%d/'%(state,epochind)
            modelname=('model_%s_%d'%(state,epochind))
            modelfolddir=('../network/'+modelname) 
            while os.path.exists(resdir):
                epochind+=1
                resdir='../res/%s_%d/'%(state,epochind)
                modelname=('model_%s_%d'%(state,epochind))
                modelfolddir=('../network/'+modelname)
            #load lastest unevaluated network if exists
            if  os.path.exists(modelfolddir):
                print ('start loading model_%s_%d'%(state,epochind)) 
                modelname=('model_%s_%d'%(state,epochind))           
                modeldir=('../network/%s/%s'%(modelname,modelname))
                saver.restore(sess,modeldir)
            else:
                evaluate=False
                print "there is no unevaluated network"       
        
        #evaluate result
        if evaluate:
            print "start evaluation %s_%d"%(state,epochind)
            resdir='../res/%s_%d/'%(state,epochind)
            testall(sess,result,number_of_classes,x, y_,keep_prob,speed,quicktest=quicktest,resdir=resdir,saveres=True)
            
if sys.argv[1] =="evaluate":
    trainEpoch(evaluate=True,train=False)
elif sys.argv[1] =="evaluaterandom":
    trainEpoch(evaluate=True,train=False,rand=True)
elif sys.argv[1] =="randomtrain":
    trainEpoch(rand=True)
else:
    trainEpoch()

print "finished"


