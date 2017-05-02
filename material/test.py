from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D,Deconvolution2D
from keras.optimizers import SGD,RMSprop,Adagrad,Adadelta,Adam,Adamax,Nadam
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing as pp
import numpy as np
import sys
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from PIL import Image
ClassNumber=2
width=512
height=512

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


def saveResult(y_predict,y_true,oneHot=False):
  if oneHot:
    y_predict=fromOneHot(y_predict)
    y_true=fromOneHot(y_true)
  if y_predict.shape[0]!=y_true.shape[0]:
    print "The size is different"
  res=np.zeros([y_predict.shape[0],2])
  for i in range(y_predict.shape[0]):
    res[i,0]=y_predict[i]
    res[i,1]=y_true[i]
  return res

def toOneHot(mat,size):
  cmat = np.zeros([mat.shape[0],size])
  for i in range(mat.shape[0]):
    cmat[i,mat[i]]=1
  return cmat.astype(int)
def fromOneHot(mat):
  mat=mat.astype(int)
  cmat = np.zeros([mat.shape[0]])
  for i in range(mat.shape[0]):
    for k in range(mat.shape[1]):
      if(mat[i][k]==1):
        cmat[i]=k
  return cmat.astype(int)

def evaRes(y_hats,y_trues,size):
    cmat = np.zeros([size+1,size+1])
    for ind in xrange(len(y_trues)):
        y_hat =y_hats[ind]
        y_true = y_trues[ind]
        cmat[y_hat, y_true] += 1
    print cmat
def eva(difference,plot=False):	
	sorted_diff=np.sort(difference)
	print "70% in ", sorted_diff[sorted_diff.size*7/10]
	print "Average difference in ",np.average(difference)
	if plot:  
		plt.step(sorted_diff,np.arange(sorted_diff.size).astype(float)/sorted_diff.size)
		plt.show()
def next_batch(pos,size,data):
	if pos+size<data.shape[0]:
		return pos+size,data[pos:pos+size]
	else:
		return pos+size-data.shape[0],np.concatenate((data[pos:],data[0:pos+size-data.shape[0]]),axis=0)

def divide_data(x):
	while 144>x.shape[1]-2:
		x=np.insert(x,x.shape[1],0,axis=1)
	print x.shape
	#resx=np.zeros([x.shape[0],12,12,1])
        resx=np.reshape(x[:,2:],[x.shape[0],12,12,1])
	label=x[:,1]
	label_min=6
	label_max=26
	one_hot=np.zeros([x.shape[0],19])
	ageDiff=np.zeros([x.shape[0],19])
	lossfunc=np.zeros([x.shape[0],19])
	for i in range(one_hot.shape[0]):
		one_hot[i,x[i,1]-label_min]=1
		ageDiff[i]+=np.arange(19)
		ageDiff[i]-=x[i,1]-label_min
		lossfunc[i,:]=np.abs(ageDiff[i,:])*(0-1)+1.5
		lossfunc[i,lossfunc[i,:]<0]=0
		lossfunc[i,lossfunc[i,:]>1]=0
	lossfunc*=np.sign(ageDiff)
	return resx,one_hot,ageDiff,lossfunc
	
class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(shape=(input_shape[1], self.output_dim),
                                 initializer='random_uniform',
                                 trainable=True)
        super(MyLayer, self).build()  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        return K.dot(x, self.W)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)
        
def VGG_19(weights_path=None):
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3,512,512)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(19, activation='softmax'))
	return model

def seg(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(512,512,3)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Convolution2D(4096, 1, 1, activation='relu')) 
    model.add(Dropout(0.5)) 
    model.add(Convolution2D(4096, 1, 1, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(ClassNumber, 1, 1, activation='relu'))

    #model.add(Dense(ClassNumber, activation='softmax'))
    #subsample
    model.add(Deconvolution2D(ClassNumber, 2, 2, output_shape=(None,32,32,ClassNumber), subsample=(2, 2),border_mode='valid'))
    model.add(Deconvolution2D(ClassNumber, 2, 2, output_shape=(None,64,64,ClassNumber), subsample=(2, 2),border_mode='valid'))
    model.add(Deconvolution2D(ClassNumber, 2, 2, output_shape=(None,128,128,ClassNumber), subsample=(2, 2),border_mode='valid'))
    model.add(Deconvolution2D(ClassNumber, 2, 2, output_shape=(None,256,256,ClassNumber), subsample=(2, 2),border_mode='valid'))
    model.add(Deconvolution2D(ClassNumber, 2, 2, output_shape=(None,512,512,ClassNumber), subsample=(2, 2),border_mode='valid'))
    if weights_path:
        model.load_weights(weights_path)
    return model
	
if __name__ == "__main__":
    sampleNum=0
    '''
	#Global Variable	
	folds=10
	mylr=0.01
 	mydecay=0
	mymomentum=0.5
	myshuffle=True

	testind=int(sys.argv[1])
	#cross valid
	flag=True
	for i in range(folds):
	  	temp=np.load("data"+str(i)+".npy")
	  	temp=temp.astype(int)
	  	if i==testind:
	    		test=temp
	  	else:
	    		if flag:
	      			train=temp
	      			flag=False
	    		else:
	      			train=np.concatenate((train,temp),axis=0)
	train_x,train_y,train_ageDiff,train_lf=divide_data(train)
	test_x,test_y,test_ageDiff,test_lf=divide_data(test)
    '''

    home='/home/local/VANDERBILT/yaoy4/Documents'
    image_filename = home+'/DeepInCyte/test/data/imgs/cat.jpg'
    annotation_filename = home+'/DeepInCyte/test/data/imgs/cat_annotation.png'
    
    testimage=load_image(image_filename).reshape(1,width,height,3)
    testlabel=load_image(annotation_filename)[:,:,:2].reshape(1,width,height,2)
    
    for i in range(2):
        testimage=np.concatenate((testimage,testimage),axis=0)
        testlabel=np.concatenate((testlabel,testlabel),axis=0)
    
    print testimage.shape,testlabel.shape
    sampleNum=testimage.shape[1]
    #optm = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    model = seg()
    optm=Adam(lr=0.0001)
    model.compile(loss='msle',optimizer=optm)
    model.fit(testimage, testlabel, batch_size=100, nb_epoch=1,shuffle=True)
    out = model.predict(testimage[0:2,:,:,:])
    print out.shape
    save_image(out[0,:,:,0],'testresult.png')
    '''
	y_scaler = pp.RobustScaler()
	train_y = y_scaler.fit_transform(np.argmax(train_y,axis=1).reshape([-1,1]))
	print train_y.shape
	model = seg()
	optm = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    	model.compile(loss='mse',optimizer=optm)
        model.fit(train_x, train_y, batch_size=100, nb_epoch=120,shuffle=myshuffle)

	print mylr,mydecay,mymomentum,myshuffle
	y_trues=np.argmax(test_y,axis=1)
	out = model.predict(test_x)
	out = out.flatten()
	predict=y_scaler.inverse_transform(out)
	difference=np.abs(predict-y_trues)
	eva(difference,plot=False)
	eva(np.abs(predict-y_trues),plot=False)	
	np.save("result"+str(testind)+".npy",saveResult(predict,y_trues))	
	'''
	
