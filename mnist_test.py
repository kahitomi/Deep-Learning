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


#init ann
# train_data = train_data[0:100]
# train_result = train_result[0:100]


#init rbm
rbm = ann.rbm(784, 3, 2, 0.5)

rbm.train(train_data[0:100])

# #init ann
# nn = ann.ann([784,30,10], 2, 0.01, 2)

# #combain
# nn.rbm(rbm)

# nn.train(train_data, train_result)
# files.saveData(nn, 'nn.db')

# # temp = nn.sim(train_data[0])

# def compare(nn, data, results):
# 	total = len(data)
# 	right = 0
# 	for i in range(total):
# 		a = nn.sim(data[i]).argmax()
# 		b = results[i].argmax()
# 		# print a, " : ", b
# 		if a == b:
# 			right += 1
# 	# error = error/total
# 	print "#################"
# 	print "Final result is: ", right, "/", total
# 	print "#################\n\n"

# compare(nn, test_data, test_result)


# # temp = nn.sim(train_data[0])
# # print "COMPARE==========="
# # print temp
# # print train_result[0]