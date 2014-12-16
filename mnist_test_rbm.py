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



# #init rbm
# rbm = ann.rbm(784, 3, 5, 0.3, 100)

# rbm.train(train_data)

# files.saveData(rbm, 'rbm-1000.db')



rbm = files.loadData('rbm.db')

im = rbm.sim(test_data[0:1])

im = np.array(im[0])
im = im.reshape(28,28)
 
fig = plt.figure()
plotwindow = fig.add_subplot(111)
plt.imshow(im , cmap='gray')
plt.show()

im = rbm.sim(test_data[0:1])

print np.array_equal(im[0], im[1])

im = np.array(im[0])
im = im.reshape(28,28)
 
fig = plt.figure()
plotwindow = fig.add_subplot(111)
plt.imshow(im , cmap='gray')
plt.show()