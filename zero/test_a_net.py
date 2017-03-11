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
#load net and do preproccessing
#model_weights='zero/a_net/base1_iter_40000.caffemodel'
model_weights = 'zero/a_net/snapshot_with_weight_v1/a_net_iter_40.caffemodel'
model_def='zero/a_net/deploy.prototxt'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

import cv2
from cv2 import cv
import os
import imghdr
import re
import numpy as np

img_dir='zero/test_a_net/test_data'
test_result_dir='zero/test_a_net/test_result'
all_img=os.listdir(img_dir)
all_img.sort()
for img in all_img:
		frame=cv2.imread('zero/test_a_net/test_data/'+img,0)
		frame=(frame-127.5)/128
		#frame=frame.transpose((2,0,1))
		net.blobs['image'].data[0,:] = frame
		"""
		with open('/home/brl/para.txt','rt') as f:
		  for line in f:
		    para=line.split(' ')
		    para=para[0:len(para)-1]
		    para = [float(i) for i in para]
		    net.blobs['fc6_shape'].data[:] = para[0:199]
		    net.blobs['fc6_exp'].data[:] = para[199:228]
        	    net.blobs['fc6_m'].data[:] = para[228:236]
		for i in range(199):
  		  print str(net.blobs['fc6_shape'].data[0,i]),
		  print ',',
		for i in range(29):
 		  print str(net.blobs['fc6_exp'].data[0,i]),
		  print ',',
		for i in range(8):
		  print str(net.blobs['fc6_m'].data[0,i]),
		  print ',',
		"""
		isomap = net.forward()['unrolling'][0,:]*255.0
		#print isomap.shape
		new_isomap = isomap.transpose((1,2,0)).astype(np.uint8)
		#cv2.imshow('isomap',new_isomap)
		#cv2.waitKey(0)
		cv2.imwrite(test_result_dir+os.path.sep+img,new_isomap)
		#break
#print isomap.shape
#isomap=isomap*255.0
#cv2.imshow("isomap",isomap.transpose((1,2,0)).astype(np.uint8))
#cv2.waitKey(0)
