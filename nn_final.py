import numpy as np
from scipy.misc import imresize

def process2(hidden_layer):
	nh=hidden_layer
	ni=64 # number of nodes in input layer
	no=3 # number of nodes in output layer

	epoch = 100 # number of epochs
	l1,l2=1.0,1.0 # Learning rates


	file_train = open('optdigits-orig.tra', 'r')

	X=[]
	Y=[]
	img = []
	digits = [4, 7, 8]
	mapping = {4 : '100', 7 : '010', 8 : '001'}

#read the data from file
	for i in range(21):
    	    file_train.readline()
#end of initial skip       
	count = 0
	z=[]

	for line in file_train:
		if count == 33:
			if y in digits:
				Y.append(y)
				z.append(Y)
				z.append('-')
				X.append(resize_data(np.array(img).astype(np.float)).tolist())
			img=[]
			count=0
		elif count==32:
			value=line.strip()
			y=float(value)
		else:
			dum_line=line.rstrip()
			l=list(dum_line)
			img.append(l)
		count=count+1

#append the last one
	if y in digits:
        	Y.append(y)
        	z.append(Y)
        	z.append('*')
        	X.append(resize_data(np.array(img).astype(np.float)).tolist())

	X = np.array(X)
	Y = np.array(Y)
	y = []
	for val in Y:
		new_list=list(mapping[val])
		y.append(new_list)
	y = np.array(y).astype(np.float)


	syn0, syn1 = fit(X, y,ni,nh,no,epoch)

	file_test = open('optdigits-orig.wdep', 'r')
	X, Y = [], []
	count = 0
	img = []
	y = 0
	for i in range(21):
		file_test.readline()
	for line in file_test:
		if count == 33:
			if y in digits:
				Y.append(y)
				z.append(Y)
				z.append('-')
				X.append(resize_data(np.array(img).astype(np.float)).tolist())
			img=[]
			count=0
		elif count==32:
			value=line.strip()
			y=float(value)
		else:
			dum_line=line.rstrip()
			l=list(dum_line)
			img.append(l)
		count=count+1

	#append the last one
	if y in digits:
        	Y.append(y)
        	z.append(Y)
        	z.append('*')
        	X.append(resize_data(np.array(img).astype(np.float)).tolist())

	X = np.array(X)

	# print X
	correct=0
	for i in range(0,len(X)):
        	val = predict(X[i], syn0, syn1)
        	# print "Predicted:",
        	# print (predicted_val(val))
        	# print "Original:",
        	# print Y[i]
        	if predicted_val(val)==int(Y[i]):
        		correct=correct+1

	print "Accuracy:",(float(correct)*100/len(X))

def sigmoid(x, deriv=False):
    if(deriv==True):
        return sigmoid(x)*(1-sigmoid(x))
    return 1.0/(1.0+np.exp(-x))

def fit(X, y,ni,nh,no,epoch):
    np.random.seed(0)
    synapse0 = np.random.random((ni,nh)) - 1
    synapse1 = np.random.random((nh,no)) - 1
    for j in range(epoch):
        l0 = X
        l1 = sigmoid(np.dot(l0, synapse0))
        l2 = sigmoid(np.dot(l1, synapse1))
        l2_error = y - l2
        
        l2_delta = l2_error*sigmoid(l2, deriv=True)
        l1_error = l2_delta.dot(synapse1.T)
        l1_delta = l1_error * sigmoid(l1, deriv=True)

        synapse1 += 0.01*l1.T.dot(l2_delta)
        synapse0 += 0.01*l0.T.dot(l1_delta)

    return synapse0, synapse1

def predict(X, synapse0, synapse1):
    l1 = sigmoid(np.dot(X, synapse0))
    l2 = sigmoid(np.dot(l1, synapse1))
    # print l2
    return np.argmax(l2)


def resize_data(image):
	M=imresize(image,(8,8),interp='bicubic').flatten()
	list_b=[]
	list_b.append(1)
	bias=np.array(list_b)
        for i in range(64):
                if M[i] != 0:
                        M[i] = 1
	return M

def predicted_val(value):
	dic={0:4,1:7,2:8}
	return dic[value]

# network settings

nh=[60,70,80,90,100] # number of nodes in hidden layer


for hiddenlayers in nh:
	print "---------------"
	print "Number of hidden layers:",hiddenlayers
	process2(hiddenlayers)


