# -*- coding: utf-8 -*-
import numpy as np
import struct
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
import data


def load_save_function(filename):
	binfile = open(filename , 'rb')
	buf = binfile.read()
	 
	index = 0
	magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)
	index += struct.calcsize('>IIII')
	BNum = numImages*numRows*numColumns
	im = struct.unpack_from('>'+str(BNum)+'B' , buf , index)
	im = np.array(im)
	im = im.reshape(numImages,numRows,numColumns)
	print 'success: '+filename

	return im

	# fig = plt.figure()
	# plotwindow = fig.add_subplot(111)
	# plt.imshow(im[1] , cmap='gray')
	# plt.show()

def load_save_label(filename):
	binfile = open(filename , 'rb')
	buf = binfile.read()

	labelNum = len(buf)

	labelNum -= 8

	# print labelNum
	 
	index = 0
	labels = struct.unpack_from('>'+str(labelNum)+'B' , buf , index);

	print 'success: '+filename

	return labels


train_data = load_save_function('train-images-idx3-ubyte');
train_result = load_save_label('train-labels-idx1-ubyte');
test_data = load_save_function('t10k-images-idx3-ubyte');
test_result = load_save_label('t10k-labels-idx1-ubyte');


print train_data.shape
train_data.tofile("mnist_train_data.db")
data.saveData(train_result, "mnist_train_result.db")
test_data.tofile("mnist_test_data.db")
data.saveData(test_result, "mnist_test_result.db")