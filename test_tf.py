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
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def conv2dtf(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
  
def conv2d(inMat,Weights):
  W_conv1 = weight_variable(Weights)
  b_conv1 = bias_variable([Weights[-1]])
  h_conv1 = tf.nn.relu(conv2dtf(inMat, W_conv1) + b_conv1)
  return h_conv1

def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
           
def bilinear_upsample_weights(factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    
    filter_size = get_kernel_size(factor)
    
    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)
    
    upsample_kernel = upsample_filt(filter_size)
    
    for i in xrange(number_of_classes):
        
        weights[:, :, i, i] = upsample_kernel
    
    return weights
    
def deconv2d(inMat,upsample_factor=2,number_of_classes=2):
  upsample_filter_np = bilinear_upsample_weight(upsample_factor,number_of_classes)
  upsample_filter_tensor = tf.constant(upsample_filter_np)
  downsampled_logits_shape = tf.shape(inMat)
  upsampled_logits = tf.nn.conv2d_transpose(inMat, upsample_filter_tensor,
                                 output_shape=upsampled_logits_shape,
                                 strides=[1, upsample_factor, upsample_factor, 1])
  return upsampled_logits
  
def densityLayer(net,insize,outsize):
  W_fc2 = weight_variable([insize, outsize])
  b_fc2 = bias_variable([outsize])
  y_conv = tf.matmul(net, W_fc2) + b_fc2

def FCN(x_image,number_of_classes=2):
  x_image = tf.reshape(x, [-1,512,512,1])
  out = onelayer(x_image,[3,3,1,64])
  out = onelayer(x_image,[3,3,64,64])
  out = max_pool_2x2(out)  #size to [-1,256,256,64]
  out = onelayer(out,[3,3,64,128])
  out = onelayer(out,[3,3,128,128])
  out = max_pool_2x2(out) #size to [-1,128,128,128]
  out = onelayer(out,[3,3,128,256])
  out = onelayer(out,[3,3,256,256])
  out = onelayer(out,[3,3,256,256])
  out = max_pool_2x2(out) #size to [-1,64,64,256]
  out = onelayer(out,[3,3,256,512])
  out = onelayer(out,[3,3,512,512])
  out = onelayer(out,[3,3,512,512])
  out = max_pool_2x2(out) #size to [-1,32,32,512]
  out = onelayer(out,[3,3,512,512])
  out = onelayer(out,[3,3,512,512])
  out = onelayer(out,[3,3,512,512])
  out = max_pool_2x2(out) #size to [-1,16,16,512]
  out = onelayer(out,[7,7,512,4096]) #size to [-1,16,16,4096]
  out = tf.nn.dropout(out, keep_prob)
  out = onelayer(out,[1,1,4096,4096]) #size to [-1,16,16,4096]
  out = tf.nn.dropout(out, keep_prob) 
  out = onelayer(out,[1,1,4096,number_of_classes]) #size to [-1,16,16,number_of_classes]
  out = deconv2d(out) #size to [-1,32,32,number_of_classes]
                                 
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
    

def breakpoint():
    assert 1==2
    return 
#Data preparation--------------------------
'''
number_of_classes=2  
home='.'
image_filename = home+'/data/imgs/cat.jpg'
annotation_filename = home+'/data/imgs/cat_annotation.png'
testx=load_image(image_filename)
testy=load_image(annotation_filename)[:,:,0]

testx=testx.reshape(1,512,512,3)
testy=testy.reshape(1,512,512,1)

for i in range(6):
  testx=np.concatenate((testx,testx),axis=0)
  testy=np.concatenate((testy,testy),axis=0)
  
testy=prepareY(testy,number_of_classes)
testx=prepareX(testx)
trainx=testx
trainy=testy
testx=testx
testy=testy 
                             
print  'data shape',trainx.shape,trainy.shape,testx.shape,testy.shape

number_of_classes=13  
home='../bigfile'
imgdir = home+'/img_3.npy'
segdir = home+'/seg_3.npy'
images=np.load(imgdir)
labels=np.load(segdir)
images = prepareX(images)
labels= prepareY(labels,number_of_classes)

seed = int(time.time())
trainx, testx, trainy, testy = train_test_split(images, labels, random_state=seed, train_size=0.9)
print  'data shape',trainx.shape,trainy.shape,testx.shape,testy.shape
'''
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


if sys.argv[3]=='quicktest':
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
tf.add_to_collection("y_conv", y_conv)
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
    if i==0:
        print "traindata: %d randomstate: %s, echo,iterations: %d,%d, gap: %d "%(len(selectorder),randomstate,epoch,iterations,gap)
    if sys.argv[3]=='quicktest':
        Y=np.load('../bigfile/testimgs.npy')
        X=np.load('../bigfile/testsegs.npy')
    else:    
        pos,sample=next_batch(pos,size,selectorder)
        imgs=mydataFetch.getdata(sample,'train','img')
        segs=mydataFetch.getdata(sample,'train','seg')
        Y=prepareY(segs,number_of_classes)
        X=prepareX(imgs)
    #print "step: ",i
    if i%gap == 0 or i==iterations-1:
        cp=correct_prediction.eval(feed_dict={x: X, y_: Y,keep_prob: 1.0})
        ce=cross_entropy.eval(feed_dict={x: X, y_: Y,keep_prob: 1.0})
        ac=np.mean(cp)
        ac2=np.mean(cp[1:])
        print("step %d, training accuracy %d, only label: %d, loss %g, time %d"%(i, ac,ac2,ce,time()-t0))
        t0 = time()
    train_step.run(feed_dict={x: X, y_: Y, keep_prob: 0.5})
del X,Y,sample,imgs,segs


#testing---------------------------


print ("----------------start saving model")
objectNum=25
viewNum=3
mydataFetch.clear()

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



modelname=('model_%d_%s'%(epoch,randomstate))
modeldir=('../network/'+modelname)  
if not os.path.exists(modeldir):
    os.makedirs(modeldir)
modeldir=(modeldir+'/'+modelname)  
print 'save model to: %s'%(modeldir)
saver0.save(sess, modeldir)
saver0.export_meta_graph(modeldir+'.meta')

    
    
print ("start testing")
resdir='../res/%s_%s_%s'%(sys.argv[1],sys.argv[2],sys.argv[3])
if not os.path.exists(resdir):
    os.makedirs(resdir)
mydataFetch.clear() 
   
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
                predict3D[startpos:startpos+size,:,:,0]=result.eval(feed_dict={x: imgs, y_: segs, keep_prob: 1.0})
            if viewInd==1:
                predict3D[:,startpos:startpos+size,:,1]=result.eval(feed_dict={x: imgs, y_: segs, keep_prob: 1.0}).transpose(1,0,2)
            if viewInd==2:
                predict3D[:,:,startpos:startpos+size,2]=result.eval(feed_dict={x: imgs, y_: segs, keep_prob: 1.0}).transpose(1,2,0)
    
    
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
    np.save(resdir+'%d.npy'%(objectInd),predict3DReal.reshape([512,512,512]))

