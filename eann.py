# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio  
import matplotlib.pyplot as plt

import ann
import data as files

#import the data
matfn=u'mnist/mnist_uint8.mat' 
data=sio.loadmat(matfn)

train_data = np.float64(data['train_x']) /255
train_result = np.float64(data['train_y'])
test_data = np.float64(data['test_x']) / 255
test_result = np.float64(data['test_y'])

#init data
train_data = train_data#[0:1000]
train_result = train_result#[0:1000]

# #init eann
# opt = {
# 	'architecture' : [784,1000,1200,10],
# 	'population' : 10,
# 	'mutationRate' : 0.0001,
# 	'error' : 0.04,
# 	'epochs' : 1000
# }
# eann = ann.eann(opt)

# eann.train(train_data, train_result)

# files.saveData(eann, 'eann-all.db')
# files.saveData(eann.w, 'eann-w-all.db')



#init ann
eann = files.loadData('eann-all.db')
eann_w = files.loadData('eann-w-all.db')
opt = {
	'architecture' : eann.architecture,
	'learningRate' : 9,
	'error' : 0.001,
	'epochs' : 50,
	'batch' : 100
}
nn = ann.ann(opt)
eann_w = np.asarray(eann_w)
for i in range(len(eann_w)):
	eann_w[i] = eann_w[i].astype(float)
nn.w = eann_w

nn.train(train_data, train_result)
# files.saveData(nn, 'eann-bp.db')

_results = nn.sim(test_data)
_results = _results.transpose()

accuracy = 0
for i in range(len(test_result)):
	if i < 20:
		print _results[i].argmax(), " : ", test_result[i].argmax()
	if _results[i].argmax() == test_result[i].argmax():
		accuracy += 1.00

print accuracy, " / ", len(test_result)

accuracy = accuracy/len(test_result)

print 'Test accuracy is ', accuracy

print _results[0]

