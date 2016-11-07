#
# Contains any functionality necessary to set up a dataset for deep-learning.
# The main function here is 'load_dataset()', which uses 'get_images()' and
# 'get_labels()' to parse the chosen dataset.
#
# ===========================================================================

import numpy as np
import scipy.misc
from PIL import Image
import csv

import os

import theano
import theano.tensor as T


'''	Creates a .png file from some image pixels. Useful for testing.
	Args:
		pixels: Raw image pixels.
		name: Name for the image file we're creating.
'''
def make_image(pixels, name):

	from PIL import Image
	img = Image.fromarray(pixels, 'RGB')
	img.save(name)
	img.show()


'''	Loads in a dataset from a given path, and segments it into different
	sets for cross-validation.
	Args:
		path: Path to the data directory.
	Returns:
		A list of tuples. Each tuple corresponds to a training set,
		where the first element is a theano shared variable
		containing raw image data, and the second is the corresponding
		label.
'''
def load_dataset(path="data/"):

	# Get a list of directories.
	directories = [
				_dir
				for _dir in os.listdir(path + 'train')
				if not _dir.startswith('.')
		      ]

	# Parse, and combine, data (images and their labels) from each directory.
	images, labels = [], []
	for directory in directories:
		data = parse_directory(path + 'train/' + directory)
		dir_images, dir_labels = zip(*data)
		images += dir_images
		labels += dir_labels

	'''	(Helper)
		parameters: data_x - A list of ndarray images.
			    data_y - A list of corresponding image labels.
		returns: theano shared variables. '''
	def shared_dataset(data_x, data_y):
		print 'theanoifying'
		print data_x.shape, data_x[0].shape, type(data_x[0])
		print len(data_y)
		
		shared_x = theano.shared(np.asarray(data_x,
			dtype=theano.config.floatX), borrow=True)
		shared_y = theano.shared(np.asarray(data_y,
			dtype=theano.config.floatX), borrow=True)

		return shared_x, T.cast(shared_y, 'int32')
	
	images = np.asarray(images)
	
	# Divide the data into trainign, testing, and validation sets.
	x_train, y_train = shared_dataset(images[:23520], labels[:23520])
	x_val, y_val	 = shared_dataset(images[23520:31374], labels[23520:31374])
	x_test, y_test	 = shared_dataset(images[31374:], labels[31374:])
	
	return [ (x_train, y_train), (x_val, y_val), (x_test, y_test) ]


'''	Parse a given directory.
	Args:
		dir_name: The name of the directory to parse.
	Returns:
		An array of tuples, where each tuple has raw image data as 
		its first element, and the corresponding label as its second.
'''
def parse_directory(dir_path):

	# Get images.
	images 		= get_images(dir_path)

	# Get class labels. Note, the csv file name is the last entry
	csv_fname 	= os.listdir(dir_path)[-1]
	csv_path 	= dir_path + '/' + csv_fname
	labels 		= get_labels(csv_path)

	if len(images) != len(labels): return -1
	return [
			(image, label)
			for image, label in zip(images, labels)
	       ]


'''	Read all images in a given directory into a specific format.
	The following article really helped:
	
	https://blog.eduardovalle.com/2015/08/25/input-images-theano/
	
	Args:
		dir_path: The path to the .ppm images.
	Returns:
		Returns an ndarray containing each images pixel data in
		a numpy array.
'''
def get_images(dir_path):

	# Image size and the number of channels. Note we are using RGB imgs.
	imagesize, nchannels = (15, 15), 3
	images = []
	for img_name in os.listdir(dir_path)[:-1]:
		img_path = dir_path + '/' + img_name
		img = Image.open(img_path, 'r')
		images.append(pre_process(img))

	# Format image array.
	img_shape = (1,) + imagesize + (nchannels,)
	all_images = [
			np.fromstring(i.tobytes(), dtype='uint8', count=-1, sep='')
			for i in images
		     ]
	all_images = [
			np.rollaxis(a.reshape(img_shape), 3, 1)
			for a in all_images
		     ]
	
	all_images = np.concatenate(all_images)
	
	print all_images.shape, len(all_images), all_images[0].shape
	return all_images


'''	Pre-Process an image. Currently this is a simple resize operation,
	however I plan to experiment with a number of data augmentation
	techniques.
	Args:
		img: Raw image pixels for pre-processing.
	Returns:
		The preprocessed image pixels.
'''
def pre_process(img):

	# GTSRB images range from 15x15 - 250x250.
	img_size = (15, 15)
	return img.resize(img_size, Image.BILINEAR)


'''	Returns a list of labels from a given .csv file.
	Args:
		csv_path: The path to the target .csv file.
	Returns:
		A list of class labels.
'''
def get_labels(csv_path):

	# Skip the csv header, and append the class ID (found at index 7)
	# from each row in the file.
	labels = []
	with open(csv_path, 'rb') as fp:
		dir_csv = csv.reader(fp, delimiter=';')
		dir_csv.next()
		for row in dir_csv:
			labels.append(row[7])
	
	return labels


# Load in the dataset.
data = load_dataset()
network_in = {
		'training': data[0],
		'validation': data[1],
		'test': data[2]
	     }

print 'Training: ', network_in['training'][0].shape.eval()
print 'Training: ', network_in['training'][1].shape.eval()
print 'Validation: ', network_in['validation'][0].shape.eval()
print 'Test: ', network_in['test'][0].shape.eval()

print network_in['training'][0][0].eval()

