#
#	Convolutional Neural Network.
#


import numpy as np

import theano
import theano.tensor as T

import lasagne as nnet

from data import *


class ConvNNet():
	
	'''	Initialize a convolutional neural network.
		We'll use the Glorot initialization for our weights.
		ReLu for non-linearity, and a 43-way Softmax for our classifier.
		N.B. -- Each layer will be have a suffix from a-z, with an
			abbreviation as the suffix.
		
			E.g. self.b_cnv == Second(b) layer is Convolutional.
		Args:
			inpt_shape: The dimensions of the input tensor.
				    (__, channels, width, height)
	'''
	def __init__(self, inpt_shape):

		
		self.inpt	= T.tensor4('inputs')
		self.y		= T.ivector('y')
		
		self.W		= nnet.init.GlorotUniform()
		
		self.nonlin 	= nnet.nonlinearities.rectify
		self.classifier = nnet.nonlinearities.softmax
		
		# Input layer.
		self.a_in  = nnet.layers.InputLayer(shape=inpt_shape,
						    inpt_var=self.inpt
		)
		
		# Convolutional Layers.
		self.b_cnv = nnet.layers.Conv2DLayer(self.a_in,
						     num_filters=32,
						     filter_size=(5,5),
						     nonlinearity=self.nonlin,
						     W=self.W
		)
		self.c_cnv = nnet.layers.Conv2DLayer(self.b_cnv,
						     num_filters=32,
						     filter_size=(5,5),
						     nonlinearity=self.nonlin
		)
		self.d_cnv = nnet.layers.Conv2DLayer(self.c_cnv,
						     num_filters=32,
						     filter_size=(5,5),
						     nonlinearity=self.nonlin
		)

		# Fully Connected / Dense layers.
		self.e_fc  = nnet.layers.DenseLayer(self.d_cnv,
						    num_units=30,
						    nonlinearity=self.nonlin
		)
	
		# Classifier.
		self.cnet = nnet.layers.DenseLayer(self.e_fc,
						    num_units=43,
						    nonlinearity=self.classifier
		)

		self.params = nnet.layers.get_all_params(self.cnet, trainable=True)
		print self.params


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
		x_test_batches	 = x_test.shape.eval()[0] / batch_size

		# Create a cost expression.
		pred 	  = nnet.layers.get_output(self.cnet)
		cost	  = (nnet.objectives.categorical_crossentropy(
				pred, self.y)).mean()
		updates	  = nnet.updates.nesterov_momentum(
					cost, self.params,
					learning_rate=eta, momentum=rho)
		
		test_pred = nnet.layers.get_output(self.cnet, determinstic=True)
		test_loss = nnet.objectives.categorical_crossentropy(pred, self.y)
		test_acc  = T.mean(T.eq(T.argmax(test_pred, axis=1), self.y),
				dtype=theano.config.floatX)

		train_model    = theano.function(inputs=[self.inpt, self.y],
						 outputs=cost,
						 updates=updates
		)
		validate_model = theano.function(inputs=[self.inpt, self.y],
						 outputs=[test_loss, test_acc]
		)

		print '\n\tbeginning training ...'
		for epoch in xrange(epochs):
			for batch_index in xrange(n_train_batches):
				x_batch = x_train[ i * batch_size: 
						  (i + 1) * batch_size ]
				y_batch = y_train[ i * batch_size:
						  (i + 1) * batch_size ]
				x = train_model(x_batch, y_batch)
			
			val_xb = x_val[:20]
			val_yb = y_val[:20]
			err, acc = validate_model(val_xb, val_yb)

# Load in the data.
print '\n\tloading data ...'
data = load_dataset()
network_in = {
		'training': data[0],
		'validation': data[1],
		'test': data[2]
	     }

print 'Training: ', network_in['training'][0].shape.eval()[0], \
		    network_in['training'][1].shape.eval()
print 'Validation: ', network_in['validation'][0].shape.eval()
print 'Test: ', network_in['test'][0].shape.eval()

print '\n\tcreating convolutional neural network.'
gtsrb_convnet = ConvNNet(inpt_shape=(None, 3, 15,15))
print '\n\tCNN initialized. beginning training ...'
BATCH_SIZE = 120
gtsrb_convnet.train(
				dataset=network_in,
				batch_size=BATCH_SIZE,
				epochs=500,
				eta=0.01,
				rho=0.9
			)

