# -*- coding: utf-8 -*-
import numpy as np
import math

import matplotlib.pyplot as plt






class eann(object):
	"""EANN model"""

	architecture = []
	layer_num = 0
	w = []
	epochs = -1
	error = 0.01
	population = 1
	mutationRate = 0.001

	def __init__(self, options):
		self.architecture = options['architecture']
		self.layer_num = len(options['architecture'])
		self.epochs = options['epochs']
		self.error = options['error']
		self.population = options['population']
		self.mutationRate = options['mutationRate']

		for i in range(0, self.layer_num-1):
			self.w.append( np.random.random_integers( 3, size=( self.architecture[i+1], self.architecture[i] ) ) - 2 )


	#node core
	def core_function(self, in_value, weights, type = 'sigm'):
		scale = 1.00
		length = 1
		if type is 'sigm' :
			temp = 1.00/( 1.00 + np.exp( - np.dot( weights, in_value ) ) )
		return temp

	#train auto encoder and ann with evolution
	def train(self, train_data, train_result):
		data = train_data
		#layer weights train
		for layer in range(0, self.layer_num-1):

			print 'To upper layer ', layer+1
			#initialise population
			population = np.random.random_integers(3, size=(self.population, self.architecture[layer+1], self.architecture[layer]))-2
			print 'initialise population successed'
			individual_length = len(self.w[layer])

			#epochs
			epochs = 0
			while 1:
				epochs += 1
				# print '- epoch ', epochs

				#evaluate population
				#and return the best individual
				fitness = []
				fitnessMax = -1
				_max = 0
				for i in range(self.population):
					#calculate fitness
					if layer is self.layer_num-2:
						#output layer
						_data = self.core_function(data.transpose(), population[i])
						_error = np.mean(np.abs(_data.transpose()-train_result))
						fitness.append(1 - _error)
					else:
						#hidden layer
						_data = self.core_function(data.transpose(), population[i])
						_data = self.core_function(_data, population[i].transpose())
						fitness.append( 1 - np.mean(np.abs(data-_data.transpose())) )
					# if fitness[i] <= 0:
					# 	print 'Fitness under 0 !!!!!!!!!!!!!!!!!!!!'
					if fitness[i] > fitnessMax:
						fitnessMax = fitness[i]
						_max = i

				# if epochs%100 == 0:
				print 'layer', layer+1,'- epoch', epochs
				print '         fitness is ', fitnessMax
				print '         mean fitness is ', np.mean(fitness)

				#finish
				if fitnessMax >= 1 - self.error:
					# print 'layer ', layer+1,' - epoch ', epochs
					# print '         fitness is ', fitnessMax
					print 'Finished\n'

					#updata data
					self.w[layer] = population[_max]
					data = self.core_function(data.transpose(), population[_max])
					data = data.transpose()
					break

				if epochs >= self.epochs and self.epochs != -1:
					print 'Finished\n'
					#updata data
					self.w[layer] = population[_max]
					data = self.core_function(data.transpose(), population[_max])
					data = data.transpose()
					break;

				#reproduce with variation
				population2 = []
				#select function
				sel_index = np.argsort(fitness)
				sel_index = np.array(sel_index[self.population/2:])
				selections = np.random.choice(sel_index, self.population*2)	
				# selections = np.random.choice(self.population, self.population*2, p=fitness/np.sum(fitness))	
				# print '                               ', fitness
				# print '                               ', selections
				#crossover
				cross_points = np.nonzero(np.random.randint(2, size=(self.population, individual_length)))
				population2 = population[selections[0:self.population]]
				population2[cross_points] = population[selections[self.population:]][cross_points]
				#mutation
				muta_points = np.random.random_sample(population2.shape)
				muta_points = muta_points < self.mutationRate
				muta_points = muta_points.astype(int)
				muta_points = np.nonzero(muta_points)
				population2[muta_points] = (np.random.randint(3, size=population2.shape)-2)[muta_points]

				#update population
				population = population2
				# break












class rbm(object):
	"""RBM model"""

	layer_size = 0
	layer_num = 0
	epoch_num = 1
	learningRate = 1
	batch = 1

	w = []
	b = []

	#init rbm
	def __init__(self, layer_size, layer_num, epoch_num = 1, learningRate = 1, batch = 10):
		self.layer_size = layer_size
		self.layer_num = layer_num
		self.epoch_num = epoch_num
		self.batch = batch
		self.learningRate = learningRate
		self.w = np.random.rand(layer_num-1, layer_size, layer_size)*2-1
		self.b = np.zeros([layer_num, layer_size], dtype=float)

	#node core
	def core_function(self, in_value, weights, bias, type = 'sigm'):
		scale = 1.00
		length = 1
		# length = len(bias)
		# print bias.shape
		# print np.dot( weights, in_value ).shape
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
		temp = np.empty([self.layer_size, self.batch])

		P = self.core_function(data, w, b)

		temp = np.random.rand(self.layer_size, self.batch)>P
		temp = temp.astype(float)

		return temp

	#up to down train function
	def rbm_down(self, data, layer):
		sample_num = len(data)	
		batch_num = sample_num/self.batch	

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

			#train rbm with samples
			for i in range(batch_num):

				v1 = np.array(data[index[(i*self.batch):((i+1)*self.batch)]])
				v1 = v1.transpose()

				h1 = self.gibbs(v1, w, np.tile(b, [self.batch,1]).transpose() )
				v2 = self.gibbs(h1, w.transpose(), np.tile(a, [self.batch,1]).transpose())
				h2 = self.core_function(v2, w, np.tile(b, [self.batch,1]).transpose())

				error = sum(sum((v1 - v2)**2))/self.layer_size/self.batch

				c1 = np.dot(h1, v1.transpose())
				c2 = np.dot(h2, v2.transpose())

				_w += self.learningRate*(c1-c2)/self.batch
				_b += self.learningRate*sum((v1-v2).transpose())/self.batch
				_a += self.learningRate*sum((h1-h2).transpose())/self.batch

				errors.append(error)


			self.w[layer-1] = w+_w/batch_num
			self.b[layer] = b+_b/batch_num
			self.b[layer-1] = a+_a/batch_num

			# print "weight ", self.w[layer-1][0][0:10]

			print "epoch ", epoch, " error: ", str(sum(errors)/sample_num)

	#train rbm
	def train(self, train_data):
		data = train_data
		# data = self.rbm_up(train_data, self.w[0], self.b[0])

		# im = np.array(data[2])
		# im = im.reshape(28,28)
		 
		# fig = plt.figure()
		# plotwindow = fig.add_subplot(111)
		# plt.imshow(im , cmap='gray')
		# plt.show()

		for layer in range(1, self.layer_num):
			print '==========\nLayer ', layer
			self.rbm_down(data, layer)

			# print self.w

			data = self.rbm_up(data, self.w[layer-1], self.b[layer])

			# print data (images)

			# im = np.array(data[555])
			# im = im.reshape(28,28)
			 
			# fig = plt.figure()
			# plotwindow = fig.add_subplot(111)
			# plt.imshow(im , cmap='gray')
			# plt.show()

			# im = np.array(data[6347])
			# im = im.reshape(28,28)
			 
			# fig = plt.figure()
			# plotwindow = fig.add_subplot(111)
			# plt.imshow(im , cmap='gray')
			# plt.show()

	def sim(self, test_data):
		data = test_data
		for layer in range(1, self.layer_num):
			data = self.rbm_up(data, self.w[layer-1], self.b[layer])
		return data












class ann(object):
	"""ANNs Model"""

	architecture = []
	layer_num = 0
	learningRate = 1
	epochs = 1
	batch = 1

	w = []
	b = []

	bp_temp = []

	error = 1

	#init ann
	def __init__(self, options):
		super(ann, self).__init__()

		self.architecture = options['architecture']
		self.layer_num = len(options['architecture'])
		self.learningRate = options['learningRate']
		self.error = options['error']
		self.epochs = options['epochs']
		self.batch = options['batch']

		# random set weights
		for i in range(0, self.layer_num-1):
			self.w.append(np.random.random( [self.architecture[i+1], self.architecture[i]] )*2-1)
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
		batch_num = sample_num/self.batch
		index = np.random.permutation(sample_num)

		# index = range(sample_num)

		#train every sample
		epochs = 1
		err = []

		while epochs <= self.epochs:
			index = np.random.permutation(sample_num)
			err = []
			print "Epochs: ", epochs, "-----"
			for i in range(batch_num):
				#select train data
				_index = index[(i*self.batch):((i+1)*self.batch)]
				data = train_data[_index]
				data = data.transpose()

				#test error
				temp = self.ff(data, self.batch)
				temp = temp.transpose()

				error = np.mean(np.abs(train_result[_index] - temp))
				# print 'error is: ', error

				#bp
				self.bp(data, train_result[_index])

				err.append(error)

			err = np.mean(err)
			#add overfitting estimation
			if err < self.error:
				break
			print "error: ", err, "\n\n"
			epochs += 1


	#feedforward function
	def ff(self, inputs, data_num):
		for i in range(self.layer_num-1):
			inputs = self.core_function(inputs, self.w[i], np.tile(self.b[i], [data_num, 1]).transpose())
			self.bp_temp[i] = inputs
		return inputs


	#back propagation function
	def bp(self, inputs, outputs):

		errors = []
		for i in range(self.layer_num-2, -1, -1):
			
			if i is 0:
				_input = inputs
			else:
				_input = self.bp_temp[i-1]

			_output = self.bp_temp[i]

			#calculate errors
			if i is self.layer_num-2:
				#hidden to output
				_input = self.bp_temp[i-1]

				_f = _output*(1 - _output)

				_error = (_output - outputs.transpose()) * _f

				errors = _error

			else:
				#hidden layers
				_error = np.dot(errors.transpose(), self.w[i+1])
				_error = _error.transpose()

				_error *= _output*(1 - _output)


			#updata
			self.b[i] -= sum(_error.transpose())*self.learningRate/self.batch
			self.w[i] -= np.dot(_error, _input.transpose())*self.learningRate/self.batch
			errors = _error
				
			# print errors


	#simulate function
	def sim(self, inputs):
		return self.ff(inputs.transpose(), len(inputs))