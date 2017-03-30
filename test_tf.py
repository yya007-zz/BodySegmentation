import tensorflow as tf
import FCN1
from PIL import Image
import numpy as np
import tensorflow as tf
import time
from scipy.misc import imsave
from sklearn.model_selection import train_test_split
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

def prepareY(y,number_of_classes):
    yf=y.flatten()
    print yf.shape
    print y.shape
    res=np.zeros([yf.shape[0],number_of_classes])
    for i in range(number_of_classes):
        res[yf[:]==i,i]=1
    return res.reshape([y.shape[0],y.shape[1],y.shape[2],number_of_classes])
#input a none,rows,cols,3 rgb image
#out put a none,rows,cols bgr image
def prepareX(rgb):
    VGG_MEAN = [103.939, 116.779, 123.68]
    res=np.zeros(rgb.shape)
    res[:,:,:,2]= rgb[:,:,:,0]-VGG_MEAN[2]
    res[:,:,:,1]= rgb[:,:,:,1]-VGG_MEAN[1]
    res[:,:,:,0]= rgb[:,:,:,2]-VGG_MEAN[0]    
    return res 
    
def next_batch(pos,size,data):
  if pos+size<data.shape[0]:
    return pos+size,data[pos:pos+size]
  else:
    return pos+size-data.shape[0],np.concatenate((data[pos:],data[0:pos+size-data.shape[0]]),axis=0)

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
'''
number_of_classes=13  
home='../bigfile'
imgdir = home+'/img.npy'
segdir = home+'/seg.npy'
images=np.load(imgdir)
labels=np.load(segdir)
images = prepareX(images)
labels= prepareY(labels,number_of_classes)

seed = int(time.time())
trainx, testx, trainy, testy = train_test_split(images, labels, random_state=seed, train_size=0.9)
print  'data shape',trainx.shape,trainy.shape,testx.shape,testy.shape


#Network structure--------------------------                                 
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None,512,512,3])
y_ = tf.placeholder(tf.float32, shape=[None,512,512,number_of_classes])
keep_prob = tf.placeholder(tf.float32)
y_conv=FCN1.FCN(x,keep_prob,number_of_classes=number_of_classes)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,3), tf.argmax(y_,3))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
result =tf.argmax(y_conv,3)
sess.run(tf.global_variables_initializer())

#training--------------------------
pos=0
size=10
for i in range(100):
  pos,X=next_batch(pos,size,trainx)
  pos,Y=next_batch(pos,size,trainy)
  print "step: ",i
  if i%10 == 0:
    ac=accuracy.eval(feed_dict={x: X, y_: Y,keep_prob: 1.0})
    ce=cross_entropy.eval(feed_dict={x: X, y_: Y,keep_prob: 1.0})
    print("step %d, training accuracy %g, loss %g"%(i, ac,ce))
    print("test accuracy %g"%accuracy.eval(feed_dict={
    x: testx, y_: testy, keep_prob: 1.0}))
  train_step.run(feed_dict={x: X, y_: Y, keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: testx, y_: testy, keep_prob: 1.0}))
testResult=result.eval(feed_dict={x: testx, y_: testy, keep_prob: 1.0})

print np.mean(testResult[0]==np.argmax(testy,axis=3)[0])
correct=(testResult[0]==np.argmax(testy,axis=3)[0])
imsave('../res/ori0.png',testx[0])
imsave('../res/prediction0.png',testResult[0])
imsave('../res/label0.png',np.argmax(testy,axis=3)[0])
imsave('../res/correct0.png',correct)

