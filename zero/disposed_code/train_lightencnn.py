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

from google.protobuf import text_format
from caffe.proto import caffe_pb2

solver=caffe.SGDSolver('zero/modified_lightencnn/solver.prototxt')
solver.net.copy_from('zero/modified_lightencnn/final_LightenedCNN_C.caffemodel')

import cv2
from cv2 import cv
import os
import imghdr
import re
import numpy as np
from sklearn import preprocessing
"""
model_weights = '/home/brl/github/unrolling/zero/what_the_fuck/final_LightenedCNN_C.caffemodel'
model_def='/home/brl/github/unrolling/zero/what_the_fuck/LightenedCNN_C_deploy.prototxt'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
		caffe.TEST)
model_weights_1 = '/home/brl/github/unrolling/zero/what_the_fuck/final_LightenedCNN_C.caffemodel'
model_def_1='/home/brl/github/unrolling/zero/what_the_fuck/LightenedCNN_C_deploy.prototxt'
net_1 = caffe.Net(model_def_1,      # defines the structure of the model
                model_weights_1,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)                caffe.TEST)     # use test mode (e.g., don't perform dropout)

layers=['conv1','conv2','conv2a','conv3','conv3a','conv4','conv4a','conv5','conv5a','fc1']
for layer in layers:
    #print net.params[layer][0].data - solver.net.params[layer+'_p'][0].data
    if (net.params[layer][0].data - solver.net.params[layer+'_p'][0].data).any() == True:
      print "core error in zero/train.py.what the fuck. the error denotes the params are not shared"
    if (net.params[layer][0].data - solver.net.params[layer][0].data).any() == True:
      print "core error in zero/train.py.what the fuck. the error denotes the params are not shared"
layer='fc1'
neuron='fc1'
"""
niter=20000
image_resize=128
for i in range(1,niter):

	#print net.params[layer][0].data[0,:]-solver.net.params[layer+'_p'][0].data[0,:]

	solver.step(1)
	print solver.net.blobs['loss'].data
	first_data=solver.net.blobs['data'].data[0,:]
	second_data=solver.net.blobs['data_p'].data[0,:]
	first_data=first_data.transpose((1,2,0))/255.0
	second_data=second_data.transpose((1,2,0))/255.0
	cv2.imshow("first",first_data)
	cv2.waitKey(1000)
	cv2.imshow("second",second_data)
	cv2.waitKey(1000)
	"""
	data=cv2.imread("/home/brl/data/unrolling/Aaron_Peirsol_0001.jpg",0)
	data_p=cv2.imread("/home/brl/data/unrolling/Aaron_Peirsol_0002.jpg",0)
	#cv2.imshow("data",data)
	#cv2.waitKey(1000)
	#cv2.imshow("data",data_p)
	#cv2.waitKey(1000)
	#data=preprocessing.normalize(data,norm='l2')
	#data_p=preprocessing.normalize(data_p,norm='l2')

	#data=cv2.resize(data,(image_resize,image_resize))
	transformed_data=np.ndarray(shape=(1,image_resize,image_resize),dtype=float,order='C')
	#print transformed_data_p.shape
	transformed_data[0,:]=data/255.0

	#data_p=cv2.resize(data_p,(image_resize,image_resize))
	transformed_data_p=np.ndarray(shape=(1,image_resize,image_resize),dtype=float,order='C')
	#print transformed_data_p.shape
	transformed_data_p[0,:]=data_p/255.0

	#print transformed_data-transformed_data_p
	net.blobs['data'].data[0,:] = transformed_data
	net_1.blobs['data'].data[0,:] = transformed_data_p
	#net.blobs['label'].data[:]=1

	net.forward()
	net_1.forward()
	print net.blobs[layer].data[0,:]-net_1.blobs[layer].data[0,:]
	#print net.blobs['data'].data[0,:]-solver.net.blobs['data_p'].data[0,:]
	#print net.blobs[layer].data[0,:]-solver.net.blobs[layer].data[0,:]
	#print net.blobs[layer].data[0,:]-solver.net.blobs[layer].data[0,:]
	#print net.blobs['loss'].data[0,:]-solver.net.blobs['loss'].data[0,:]
	break
	
	layers=['conv1']
	for layer in layers:
 	  #print net.params[layer][0].data - solver.net.params[layer+'_p'][0].data
   	  #if (net.blobs[layer].data - solver.net.blobs[layer+'_p'].data).any() == True:
      	  #  print "core error in zero/train.py.what the fuck. the error denotes the params are not shared"
          if (net.blobs[layer].data - solver.net.blobs[layer].data).any() == True:
            print "why why why"
	break
	#eltwise_fc1=net.blobs['eltwise_fc1'].data
	#eltwise_fc1_p=net.blobs['eltwise_fc1_p'].data
	#print solver.net.blobs['loss'].data
	
        if i%1==0:
	  layers=['conv1','conv2','conv2a','conv3','conv3a','conv4','conv4a','conv5','conv5a','fc1']
	  for layer in layers:
	    if (solver.net.params[layer+'_1'][0].data - solver.net.params[layer+'_p'][0].data).any() == True:
		print "core error in zero/train.py.what the fuck. the error denotes the params are not shared"


	
	layers=['conv1','conv2']
	for layer in layers:
	  print net.blobs[layer+'_1'].data - net.blobs[layer+'_p'].data
	"""

	#cv2.imshow("asf",transformed_data[0,:]/255)
	#cv2.waitKey(0)



