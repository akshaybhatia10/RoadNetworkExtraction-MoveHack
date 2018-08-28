import os
import sys
sys.path.append('./fastai')
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from fastai.conv_learner import *
from fastai.dataset import *

def convert_satellite_img(old_path, new_path, i):
    input_file = '{}/{}.tiff'.format(old_path, i)
    output_file = '{}/{}.png'.format(new_path, i)
    Image.open(input_file).resize((1024,1024)).save(output_file)

def convert_mask_img(old_path, new_path, i):
    input_file = '{}/{}.tif'.format(old_path, i)
    output_file = '{}/{}.png'.format(new_path, i)
    Image.open(input_file).resize((1024,1024)).save(output_file)

def convert_and_resize(base_path='mass_roads', new_base_path = 'mass_roads_new'
):
	
	print ('Setting up new directories')
	new_base_path = 'mass_roads_new'
	if not os.path.isdir(new_base_path):
		os.mkdir(new_base_path)

	#New Dataset Paths
	train_path_new = os.path.join(new_base_path, "train")
	valid_path_new = os.path.join(new_base_path, "valid")
	test_path_new = os.path.join(new_base_path, "test")
	#Check if new directories exist or not
	for d in (train_path_new, valid_path_new, test_path_new):
		if not os.path.isdir(d):
			os.mkdir(d)

	train_satellite_new = os.path.join(new_base_path, "train/sat")
	train_mask_new = os.path.join(new_base_path, "train/map")
	valid_satellite_new = os.path.join(new_base_path, "valid/sat")
	valid_mask_new = os.path.join(new_base_path, "valid/map")
	test_satellite_new = os.path.join(new_base_path, "test/sat")
	test_mask_new = os.path.join(new_base_path, "test/map")
	#Check if new directories exist or not
	for d in (train_satellite_new, train_mask_new, valid_satellite_new, valid_mask_new, test_satellite_new, test_mask_new):
		if not os.path.isdir(d):
			os.mkdir(d) 
	
	#Current Dataset Paths
	train_path = os.path.join(base_path, "train")
	valid_path = os.path.join(base_path, "valid")
	test_path = os.path.join(base_path, "test")

	train_satellite = os.path.join(base_path, "train/sat")
	train_mask = os.path.join(base_path, "train/map")
	valid_satellite = os.path.join(base_path, "valid/sat")
	valid_mask = os.path.join(base_path, "valid/map")
	test_satellite = os.path.join(base_path, "test/sat")
	test_mask = os.path.join(base_path, "test/map")

	train_satellite_files = glob(os.path.join(train_satellite, "*.tiff"))
	train_satellite_ids = [s[len(train_satellite)+1:-5] for s in train_satellite_files]
	valid_satellite_files = glob(os.path.join(valid_satellite, "*.tiff"))
	valid_satellite_ids = [s[len(valid_satellite)+1:-5] for s in valid_satellite_files]
	test_satellite_files = glob(os.path.join(test_satellite, "*.tiff"))
	test_satellite_ids = [s[len(test_satellite)+1:-5] for s in test_satellite_files]

	train_mask_files = glob(os.path.join(train_mask, "*.tif"))
	train_mask_ids = [s[len(train_mask)+1:-5] for s in train_mask_files]
	valid_mask_files = glob(os.path.join(valid_mask, "*.tif"))
	valid_mask_ids = [s[len(valid_mask)+1:-5] for s in valid_mask_files]
	test_mask_files = glob(os.path.join(test_mask, "*.tif"))
	test_mask_ids = [s[len(test_mask)+1:-5] for s in test_mask_files]
	print ('Done')

	print ('Converting to PNG')
	#Convert Satellite(Input) images to PNG
	for i in train_satellite_ids:
		convert_satellite_img(train_satellite, train_satellite_new, i)
	for i in valid_satellite_ids:
		convert_satellite_img(valid_satellite, valid_satellite_new, i)
	for i in test_satellite_ids:
		convert_satellite_img(test_satellite, test_satellite_new, i)

	#Convert Mask(Output) images to PNG
	for i in train_satellite_ids:
		convert_mask_img(train_mask, train_mask_new, i)
	for i in valid_satellite_ids:
		convert_mask_img(valid_mask, valid_mask_new, i)
	for i in test_satellite_ids:
		convert_mask_img(test_mask, test_mask_new, i)
	print ('Done')

class MatchedFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y=y
        assert(len(fnames)==len(y))
        super().__init__(fnames, transform, path)
    
    def get_y(self, i): 
    	return open_image(os.path.join(self.path, self.y[i]))
    
    def get_c(self): 
    	return 0
