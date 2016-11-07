import numpy as np
import scipy.misc
from PIL import Image
import csv

import os

import theano
import theano.tensor as T


def make_image(pixels, name):

	from PIL import Image
	img = Image.fromarray(pixels, 'RGB')
	img.save(name)
	img.show()


'''	parameters:
		path - Path to the data directory.
	returns:
		x - An array of image (x) and label (y) pairs. '''
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
		# TODO: (# of images, image width x height)
		# TODO: (# of labels, )
		shared_x = theano.shared(data_x, borrow=True)
		shared_y = theano.shared(np.asarray(data_y,
			dtype=theano.config.floatX), borrow=True)

		return shared_x, T.cast(shared_y, 'int32')
	
	print '\n\nIMPORTANT\t\t', type(images), type(labels)
	images = np.asarray(images)
	print images[0].shape
	# Divide the data into trainign, testing, and validation sets.
	x_train, y_train = shared_dataset(images[:10], labels[:10])
	#x_train, y_train = shared_dataset(images[:23520], labels[:23520])
	#x_val, y_val	 = shared_dataset(images[23520:31374], labels[23520:31374])
	#x_test, y_test	 = shared_dataset(images[31374:], labels[31374:])
	print x_train.eval(), y_train
	return (1,2)
	#return [ (x_train, y_train), (x_val, y_val), (x_test, y_test) ]


'''	parameters:
		dir_name - The name of the directory to parse.
	returns:
		tuple - (list of images in ndarrays, csv file handle) '''
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


'''	parameters:
		dir_path - The path to the .ppm images.
	returns:
		list - Each image in the directory in ndarray foramt. '''
def rget_images(dir_path):

	images = []
	for img_name in os.listdir(dir_path)[:-1]:
		img_path = dir_path + '/' + img_name
		images.append(scipy.misc.imread(img_path))

	return images

def get_images(dir_path):

	imagesize = (15, 15)
	nchannels = 3
	i = 0
	images = []
	for img_name in os.listdir(dir_path)[:-1]:
		img_path = dir_path + '/' + img_name
		img = Image.open(img_path, 'r')
		images.append(preprocess(img))

	ImageShape = (1,) + imagesize + (nchannels,)
	all_images = [
			np.fromstring(i.tobytes(), dtype='uint8', count=-1, sep='')
			for i in images
		     ]
	all_images = [
			np.rollaxis(a.reshape(ImageShape), 3, 1)
			for a in all_images
		     ]
	
	all_images = np.concatenate(all_images)
	
	print all_images.shape, len(all_images), all_images[0].shape
	return all_images


'''	parameters:
		img - Raw image pixels for pre-processing.
	returns:
		list - The preprocessed image pixels. '''
def preprocess(img):

	# GTSRB images range from 15x15 - 250x250.
	img_size = (15, 15)
	return img.resize(img_size, Image.BILINEAR)


'''	parameters:
		csv_path - The path to the .csv file containing labels.
	returns:
		array - A list of class labels. '''
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

#TODO: Read these articles:
# https://blog.eduardovalle.com/2015/08/25/input-images-theano/
imgs, labels = load_dataset()

