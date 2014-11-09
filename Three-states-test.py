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


#load nn
nn = files.loadData('nn.db')

# _w = nn.w
# _w = np.absolute(_w)
# avr = 0
# for i in _w:
# 	avr += np.mean(i)
# avr = avr/len(_w)

# for i in range(len(_w)):
# 	for j in range(len(_w[i])):
# 		for n in range(len(_w[i][j])):
# 			temp = _w[i][j][n]

# 			if temp > avr:
# 				temp = 1
# 			elif temp<avr and temp>-avr:
# 				temp = 0
# 			else:
# 				temp = -1

# 			_w[i][j][n] = temp

# nn.w = _w



#test
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