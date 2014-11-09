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



# nn = ann.ann([3,5,2], 10)
# nn.train([[0, 0.5, 0.2]], [[0.8, 0.2]])

# temp = nn.sim([0, 0.5])


#init data
train_data = train_data#[0:100]
train_result = train_result#[0:100]

# test_data = train_data#[[0, 19]]
# test_result = train_result#[[0, 19]]

# sample_num = len(train_data)
# index = np.random.permutation(sample_num)

# print index


# _index = index[0:2]

# print _index

# print train_data[_index].shape




#init ann
opt = {
	'architecture' : [784,30,10],
	'learningRate' : 0.3,
	'error' : 0.001,
	'epochs' : 100,
	'batch' : 100
}
nn = ann.ann(opt)

#combain
rbm = files.loadData('rbm.db')
nn.rbm(rbm)

#train
nn.train(train_data, train_result)
files.saveData(nn, 'nn.db')

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







# temp = nn.sim(train_data[0])
# print "COMPARE==========="
# print temp
# print train_result[0]


# im = np.array(im)
# im = im.reshape(28,28)
 
# fig = plt.figure()
# plotwindow = fig.add_subplot(111)
# plt.imshow(im , cmap='gray')
# plt.show()