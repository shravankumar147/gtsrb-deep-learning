#
#	Convolutional Neural Network.
#


import numpy as np

import theano
import theano.tensor as T

import lasagne as nnet

import cPickle as pickle

from data import *


class ConvNNet():
	
	'''	Initialize a convolutional neural network.
		We'll use the Glorot initialization for our weights.
		ReLu for non-linearity, and a 43-way Softmax for our classifier.
		Args:
			inpt_shape: The dimensions of the input tensor.
				    (__, channels, width, height)
			use_existing: Toggle to use existing model params.
	'''
	def __init__(self, inpt_shape, use_existing=False):
		
		self.inpt	= T.tensor4('inputs')
		self.y		= T.ivector('y')
			
		# Architecture inspired by LeNet.
		l0 = nnet.layers.InputLayer(shape=inpt_shape,
						 input_var=self.inpt)
		l1 = nnet.layers.Conv2DLayer(
				l0, num_filters=32, filter_size=(3,3),
				nonlinearity=nnet.nonlinearities.rectify,
				W=nnet.init.GlorotUniform())
		l2 = nnet.layers.MaxPool2DLayer(l1, pool_size=(2,2))
		l3 = nnet.layers.Conv2DLayer(
				l2, num_filters=32, filter_size=(3,3),
				nonlinearity=nnet.nonlinearities.rectify)
		l4 = nnet.layers.MaxPool2DLayer(l3, pool_size=(2,2))
		l5 = nnet.layers.DenseLayer(
				nnet.layers.dropout(l4, p=0.5),
				num_units=100,
				nonlinearity=nnet.nonlinearities.rectify)
		self.cnet = nnet.layers.DenseLayer(
				nnet.layers.dropout(l5, p=0.5),
				num_units=43, 
				nonlinearity=nnet.nonlinearities.softmax)
	
		# Load in parameters from an existing model.
		if use_existing:
			print '\tloading existing model.'
			with np.load('gtsrb-model.npz') as fp:
				params = [fp['arr_%d' % i]
						for i in range(len(fp.files))]
				nnet.layers.set_all_param_values(self.cnet, params)

		self.params = nnet.layers.get_all_params(self.cnet, trainable=True)

	'''	Train the model with the given parameters.
		Args:
			eta: The learning rate.
			rho: The momentum factor.
			epochs: The number of epochs to train for.
	'''
	def train(self, dataset, batch_size, epochs, eta, rho):

		# Read in the dataset, and calculate the batch size.
		x_train, y_train = dataset['training']
		x_val, y_val	 = dataset['validation']
		x_test, y_test	 = dataset['test']
		
		n_train_batches  = x_train.shape.eval()[0] / batch_size
		n_val_batches	 = x_val.shape.eval()[0] / batch_size
		n_test_batches	 = x_test.shape.eval()[0] / batch_size
			
		# Create a cost expression.
		prediction = nnet.layers.get_output(self.cnet)
		loss = nnet.objectives.categorical_crossentropy(
				prediction, self.y)
		loss = loss.mean()	
		
		# Update rule. We'll use Nesterov Momentum to speed up convergence.
		updates = nnet.updates.nesterov_momentum(loss, self.params,
				learning_rate=eta, momentum=rho)
	
		test_pred = nnet.layers.get_output(self.cnet, deterministic=True)
		test_loss = nnet.objectives.categorical_crossentropy(
				test_pred, self.y)
		test_loss = test_loss.mean()
		test_acc = T.mean(T.eq(T.argmax(test_pred, axis=1), self.y),
				dtype=theano.config.floatX)
		
		# Definitions for training and validating our model.
		train_model = theano.function(
				[self.inpt, self.y], loss, updates=updates
		)
		val_model = theano.function(
				[self.inpt, self.y], [test_loss, test_acc]
		)

		print '\n\tbeginning training ...'
		for epoch in xrange(epochs):
			
			train_err, train_batches = 0, 0
			for batch_index in xrange(n_train_batches):
				x_batch = x_train[batch_index * batch_size: 
						  (batch_index + 1) * batch_size]
				y_batch = y_train[batch_index * batch_size:
						  (batch_index + 1) * batch_size]
				train_err += train_model(
						x_batch.eval(), y_batch.eval())
				train_batches += 1

			val_err, val_acc, val_batches = 0, 0, 0
			val_avg = 0
			for batch_index in xrange(n_val_batches):
				x_batch = x_val[batch_index * batch_size:
						(batch_index + 1) * batch_size]
				y_batch = y_val[batch_index * batch_size:
						(batch_index + 1) * batch_size]
				err, acc = val_model(
						x_batch.eval(), y_batch.eval())
				val_avg += err

				val_err += err
				val_acc += acc
				val_batches += 1
			
			print '\nEpoch ', epoch, 'of ', epochs,\
				'\n\ttraining error: ',\
					train_err/train_batches,\
			       '\n\tvalidation error: ',\
			       		val_err/val_batches,\
			       '\n\tvalidation loss: ',\
			       		val_acc/val_batches * 100
			print 'Accuracy: ', (1 - val_acc/val_batches) * 100, '%'
			
			np.savez('gtsrb-model.npz', *nnet.
						     layers.
						     get_all_param_values(
							     self.cnet))


# Load in the data.
data = load_dataset()
network_in = {
		'training': data[0],
		'validation': data[1],
		'test': data[2]
	     }

# Create the Convn Net.
gtsrb_convnet = ConvNNet(inpt_shape=(None, 3, 15,15), use_existing=False)
#gtsrb_convnet = ConvNNet(inpt_shape=(None, 3, 15,15), use_existing=True)
BATCH_SIZE = 1120

# N.B. -- The model overfits quite significantly.
gtsrb_convnet.train(
				dataset=network_in,
				batch_size=BATCH_SIZE,
				epochs=300,
				eta=0.0001,
				rho=0.9
			)

