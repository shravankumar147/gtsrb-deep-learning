import numpy as np
import scipy.misc
import csv

import os


'''	parameters:
		path - Path to the data directory.
	returns:
		x - An array of image (x) and label (y) pairs. '''
def load_dataset(path="data/"):

	# Get a list of directories.
	directories = [
				image
				for image in os.listdir(path + 'train')
		      ]

	dir_data = []
	for directory in directories:
		print directory
		#print parse_directory(path + 'train/' + directory)[10]
		data = parse_directory(path + 'train/' + directory)
		dir_data.append(data)
	
	'''	(Helper)
		parameters: data - A list of image ndarray, label, pairs.
		returns: theano shared variables. '''
	def shared_dataset(data):
		
		data_x, data_y = data


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
def get_images(dir_path):

	images = []
	for img_name in os.listdir(dir_path)[:-1]:
		img_path = dir_path + '/' + img_name
		images.append(scipy.misc.imread(img_path))
	
	return images


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

load_dataset()

