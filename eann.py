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
train_data = train_data[0:1000]
train_result = train_result[0:1000]

#init eann
opt = {
	'architecture' : [784,1000,1000,10],
	'population' : 10,
	'mutationRate' : 0.0001,
	'error' : 0.04,
	'epochs' : 2
}
eann = ann.eann(opt)

eann.train(train_data, train_result)

files.saveData(eann, 'eann.db')