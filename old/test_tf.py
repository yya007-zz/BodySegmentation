import tensorflow as tf


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(inMat,Weights):
  W_conv1 = weight_variable(Weights)
  b_conv1 = bias_variable([Weights[-1]])
  h_conv1 = tf.nn.relu(conv2d(inMat, W_conv1) + b_conv1)
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

def vgg19(x_image,number_of_classes=2):
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
                                 
                                 
                                 
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)





cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

