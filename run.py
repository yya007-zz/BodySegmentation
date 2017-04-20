import tensorflow as tf
import numpy as np
import FCN1
from time import time
import os
from test_bd import dataFetch
from evaluate import testall,next_batch,prepareX,prepareY



def savemodel(modelname,saver,sess):
    print ("----------------start saving model")
    modeldir=('../network/'+modelname)  
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    modeldir=(modeldir+'/'+modelname)  
    print 'save model to: %s'%(modeldir)
    saver.save(sess, modeldir)



def trainEpoch(evaluate=False,restore=True,save=True):
    objectNum=75
    viewNum=3
    selectorder=np.arange(objectNum*viewNum*512)

    #usr define
    state="run"
    state="quicktest"
    rand=False


    size=16
    quicktest=False
    storelength=30

    if state=='quicktest':
        quicktest=True
        selectorder=np.arange(0,objectNum*viewNum*512,viewNum*512)
        selectorder=selectorder+2*512+256
        storelength=4


        
    if rand:
        selectorder=randomshuffle(selectorder)
        state="random"+state
    else:
        state="norandom"+state

    iterationsOne=len(selectorder)/size
    gap=iterationsOne//10
    mydataFetch=dataFetch(storelength)
    if gap<1:
        gap=1
    number_of_classes=19
    speed=1e-5    

    #Network structure-------------------------- 
    print "----------------start building network"                                
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
    saver = tf.train.Saver()


    if quicktest:
        imgs=np.load('../bigfile/testimgs.npy')
        segs=np.load('../bigfile/testsegs.npy')
    print "traindata: %d state: %s,iterations%d, gap: %d "%(len(selectorder),state,iterationsOne,gap)
    print "----------------start training"
    with tf.Session() as sess:
        t0 = time()   
        sess.run(tf.global_variables_initializer())
        
        
        epochind=0
        modelname=('model_%s_%d'%(state,epochind))
        modelfolddir=('../network/'+modelname)  
        while os.path.exists(modelfolddir):
            epochind+=1
            modelname=('model_%s_%d'%(state,epochind))
            modelfolddir=('../network/'+modelname)
        if  epochind>0:
            print ('start loading model_%s_%d'%(state,epochind-1)) 
            modelname=('model_%s_%d'%(state,epochind-1))           
            modeldir=('../network/%s/%s'%(modelname,modelname))
            saver.restore(sess,modeldir)
            if evaluate:
                print "start evaluation"
                resdir='../res/%s_%d/'%(state,epochind-1)
                testall(sess,result,number_of_classes,x,y_,keep_prob,quicktest=quicktest,resdir=resdir,saveres=True)
            else:
                print "no model %s"%(modelname)
        else:
            modelname=('model_%s_%d'%(state,epochind))
            print "need new model",modelname
            for iterind in range(iterationsOne):
                pos=0
                if not quicktest:
                    pos,sample=next_batch(pos,size,selectorder)    
                    imgs=mydataFetch.getdata(sample,'train','img')
                    segs=mydataFetch.getdata(sample,'train','seg')
                    imgs=prepareX(imgs)
                    segs=prepareY(segs,number_of_classes)
                train_step.run(feed_dict={x: imgs, y_: segs, keep_prob: 0.5}) 
                if iterind%gap == 0 or iterind==iterationsOne-1:
                    cp=correct_prediction.eval(feed_dict={x: imgs, y_: segs,keep_prob: 1.0})
                    ce=cross_entropy.eval(feed_dict={x: imgs, y_: segs,keep_prob: 1.0})
                    ac=np.mean(cp)
                    ac2=np.mean(cp[1:])
                    print("epoch: %d,step: %d, training accuracy %.4f, only label: %.4f, loss %g, time %d"%(epochind,iterind, ac,ac2,ce,time()-t0))
                    t0 = time()    
            if save:
                savemodel(modelname,saver,sess)
                epochind=epochind+1
                print "successfully save model"
            
        

trainEpoch()
assert 1==2
trainEpoch(evaluate=True)
print "finished"


