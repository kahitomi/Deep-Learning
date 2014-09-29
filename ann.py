# -*- coding: utf-8 -*-
import numpy as np
import math

class rbm(object):
	"""RBM model"""

	layer_size = 0
	layer_num = 0
	epoch_num = 1
	learningRate = 1

	w = []
	b = []

	#init rbm
	def __init__(self, layer_size, layer_num, epoch_num = 1, learningRate = 1):
		self.layer_size = layer_size
		self.layer_num = layer_num
		self.epoch_num = epoch_num
		self.learningRate = learningRate
		self.w = np.random.rand(layer_num-1, layer_size, layer_size)
		self.b = np.random.rand(layer_num, layer_size)

	#node core
	def core_function(self, in_value, weights, bias, type = 'sigm'):
		scale = 1.00
		length = len(bias)
		if type is 'sigm' :
			temp = scale/( scale + scale*np.exp( - bias - np.dot( weights, in_value )/length ) )
		# print temp
		return temp

	#down to up function
	def rbm_up(self, data, w, b):
		data_size = len(data)
		# temp = np.empty()

		for i in range(data_size):
			data[i] = self.core_function(data[i], w, b)

		return data

	#gibbs function
	def gibbs(self, data, w, b):
		temp = np.empty(self.layer_size)
		# for i in range(self.layer_size):
		# 	P = self.core_function(data[i], w[i], b[i])
		# 	if np.random.rand(1) > P :
			# 	temp[i] = 1
			# else:
			# 	temp[i] = 0

		P = self.core_function(data, w, b)

		for i in range(self.layer_size):
			if np.random.rand(1) > P[i] :
				temp[i] = 1
			else:
				temp[i] = 0

		return temp

	#up to down train function
	def rbm_down(self, data, layer):
		sample_num = len(data)		

		for epoch in range(self.epoch_num):
			#random train samples
			index = np.random.permutation(sample_num)
			w = self.w[layer-1]
			b = self.b[layer]
			a = self.b[layer-1]
			errors = []
			_w = np.empty((self.layer_size, self.layer_size))
			_b = np.empty(self.layer_size)
			_a = np.empty(self.layer_size)

			#train rbm with one sample
			for i in range(sample_num):

				v1 = np.array(data[index[i]])
				h1 = self.gibbs(v1, w, b)
				v2 = self.gibbs(h1, w.transpose(), a)
				h2 = self.core_function(v2, w, b)

				error = sum((v1 - v2)**2)/self.layer_size

				h1.shape = (1, self.layer_size)
				h2.shape = (1, self.layer_size)
				v1.shape = (1, self.layer_size)
				v2.shape = (1, self.layer_size)

				c1 = np.dot(h1.transpose(), v1)
				c2 = np.dot(h2.transpose(), v2)

				_w += self.learningRate*(c1-c2)
				_b += self.learningRate*(v1-v2)[0]
				_a += self.learningRate*(h1-h2)[0]

				errors.append(error)


			self.w[layer-1] = w+_w/sample_num
			self.b[layer] = b+_b/sample_num
			self.b[layer-1] = a+_a/sample_num

			# print "weight ", self.w[layer-1][0][0:10]

			print "epoch ", epoch, " error: ", str(sum(errors)/sample_num)

	#train rbm
	def train(self, train_data):
		data = train_data
		# data = self.rbm_up(train_data, self.w[0], self.b[0])
		for layer in range(1, self.layer_num):
			print '==========\nLayer ', layer
			self.rbm_down(data, layer)

			# print self.w

			data = self.rbm_up(data, self.w[layer-1], self.b[layer])

			# print data


class ann(object):
	"""ANNs Model"""

	architecture = []
	layer_num = 0
	learningRate = 1
	epochs = 1

	w = []
	b = []

	bp_temp = []

	error = 1

	#init ann
	def __init__(self, architecture, learningRate = 2, error = 0.05, epochs = 1):
		super(ann, self).__init__()

		self.architecture = architecture
		self.layer_num = len(architecture)
		self.learningRate = learningRate
		self.error = error
		self.epochs = epochs

		# random set weights
		for i in range(0, self.layer_num-1):
			self.w.append(np.random.random( [self.architecture[i+1], self.architecture[i]] ))
			self.b.append(np.zeros( self.architecture[i+1], dtype=float ))
			self.bp_temp.append(np.zeros( self.architecture[i+1], dtype=float ))


		print 'ANNs has been set up successfully'

	def rbm(self, rbm):
		self.layer_num += rbm.layer_num-1
		for i in range(rbm.layer_num-1):
			self.architecture.insert(0, rbm.layer_size)
		temp = []
		for i in rbm.w:
			temp.append(i)
		self.w = temp+self.w
		temp = []
		for i in range(1, len(rbm.b)):
			temp.append(rbm.b[i])
		self.b = temp+self.b

		self.bp_temp = []
		for i in range(0, self.layer_num-1):
			self.bp_temp.append(np.zeros( self.architecture[i+1], dtype=float ))

	def core_function(self, in_value, weights, bias, type = 'sigm'):
		scale = 1.00
		if type is 'sigm' :
			temp = scale/( scale + scale*np.exp( - bias - np.dot( weights, in_value ) ) )
		# print temp
		return temp

	#train function
	def train(self, train_data, train_result):
		sample_num = len(train_data)
		index = np.random.permutation(sample_num)

		# index = range(sample_num)

		#train every sample
		epochs = 1
		err = []

		while epochs <= self.epochs:
			err = []
			print "Epochs: ", epochs, "-----"
			for i in index:
				#add noise here

				#test error
				temp = self.ff(train_data[i])
				error = np.mean(np.abs(train_result[i] - temp))
				# print 'error is: ', error

				#bp
				self.bp(train_data[i], train_result[i])
				err.append(error)

			err = np.mean(err)
			#add overfitting estimation
			if err < self.error:
				break
			print "error: ", err, "\n\n"
			epochs += 1


	#feedforward function
	def ff(self, inputs):
		for i in range(self.layer_num-1):
			inputs = self.core_function(inputs, self.w[i], self.b[i])
			self.bp_temp[i] = inputs
		return inputs


	#back propagation function
	def bp(self, inputs, outputs):
		# print 'bp-----'

		errors = []
		for i in range(self.layer_num-2, -1, -1):
			# print '=='
			# print i

			

			if i is self.layer_num-2:
				#hidden to output
				_errors = []
				for j in range(len(self.bp_temp[i])):
					_f = self.bp_temp[i][j]*(1-self.bp_temp[i][j])
					# print _f
					if _f == 0:
						_error = (outputs[j]-self.bp_temp[i][j])*0.2
					else:
						_error = _f*(outputs[j]-self.bp_temp[i][j])
					_errors.append(_error)
					self.b[i][j] += _error*self.learningRate

				errors = _errors
				for j in range(len(self.w[i])):
					for p in range(len(self.w[i][j])):
						self.w[i][j][p] += errors[j]*self.learningRate*self.bp_temp[i-1][p]
				# print self.b[i]
				# print self.w[i]
			else:
				#hidden layers
				_output = self.bp_temp[i]
				if i is 0:
					_input = inputs
				else:
					_input = self.bp_temp[i-1]

				#calculate errors for each node
				_errors = []
				for j in range(len(_output)):
					_error = 0
					for p in range(len(errors)):
						_error += errors[p]*self.w[i+1][p][j]
					if _output[j]*(1-_output[j]) != 0:
						_error *= _output[j]*(1-_output[j])
					_errors.append(_error)
					#fine turn bias
					self.b[i][j] += self.learningRate*_error

				#fine turn weights
				for j in range(len(self.w[i])):
					for p in range(len(self.w[i][j])):
						self.w[i][j][p] += self.learningRate*_errors[j]*_input[p]

				errors = _errors
			# print errors


	#simulate function
	def sim(self, inputs):
		return self.ff(inputs)