#!/usr/bin/env python
import numpy as np
import timeit
from time import clock
import sys,os
#find the directory of the script
def cur_file_dir():
  path = sys.path[0]
  if os.path.isdir(path):
    return path
  elif os.path.isfile(path):
    return os.path.dirname(path)
def cur_file_father_dir():
  father_dir = ''
  cur_file_dir_list = cur_file_dir().split('/')
  for i in range(-len(cur_file_dir_list),-1):
    father_dir += cur_file_dir_list[i] + '/'
  return father_dir


# Make sure that caffe is on the python path:
caffe_root = cur_file_father_dir()  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_cpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2
"""
#load net and do preproccessing
model_weights = '/home/brl/github/unrolling/zero/tools/LightenedCNN_C.caffemodel'
model_def='/home/brl/github/unrolling/zero/tools/new_LightenedCNN_C_deploy.prototxt'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
"""
solver=caffe.SGDSolver('zero/tools/solver.prototxt')
solver.net.copy_from('zero/tools/LightenedCNN_C.caffemodel')

import cv2
from cv2 import cv
import os
import imghdr
import re
import numpy as np

train_image_dir="/home/brl/github/unrolling/zero/test_unrolling_layer/test_sample/IBUG"

niter=200
image_resize=128
for i in range(1,10000):

	data=cv2.imread("/home/brl/github/unrolling/zero/modified_lightencnn/001_01_01_051_03.png",0)
	data_p=cv2.imread("/home/brl/github/unrolling/zero/modified_lightencnn/Abdel_Madi_Shabneh_0001.jpg",0)
	#cv2.imshow("data",data)
	#cv2.waitKey(1000)
	#cv2.imshow("data",data_p)
	#cv2.waitKey(1000)

	data=cv2.resize(data,(image_resize,image_resize))
	transformed_data=np.ndarray(shape=(1,image_resize,image_resize),dtype=float,order='C')
	#print transformed_data_p.shape
	transformed_data[0,:]=data/255.0


	data_p=cv2.resize(data_p,(image_resize,image_resize))
	transformed_data_p=np.ndarray(shape=(1,image_resize,image_resize),dtype=float,order='C')
	#print transformed_data_p.shape
	transformed_data_p[0,:]=data_p/255.0

	#print transformed_data-transformed_data_p
	solver.net.blobs['data'].data[0,:] = transformed_data
	solver.net.blobs['data_p'].data[0,:] = transformed_data_p
	solver.net.blobs['label'].data[:]=1
	#print net.blobs['data'].data[0,:]-net.blobs['data_p'].data[0,:]

	solver.step(10)

	#eltwise_fc1=net.blobs['eltwise_fc1'].data
	#eltwise_fc1_p=net.blobs['eltwise_fc1_p'].data
	print solver.net.blobs['loss'].data
	"""
	layers=['conv1','conv2']
	for layer in layers:
	  print net.blobs[layer+'_1'].data - net.blobs[layer+'_p'].data

	layers=['conv4','conv4a','conv5','conv5a','fc1']
	for layer in layers:
	  print net.params[layer+'_1'][0].data - net.params[layer+'_p'][0].data 
	"""

	#cv2.imshow("asf",transformed_data[0,:]/255)
	#cv2.waitKey(0)



