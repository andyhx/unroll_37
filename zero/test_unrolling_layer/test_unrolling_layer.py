#!/usr/bin/env python
import numpy as np
import timeit
from time import clock
import sys,os
caffe_root='../..'
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

#load net and do preproccessing
model_weights = 'zero/test_unrolling_layer/unrolling.caffemodel'
model_def='zero/test_unrolling_layer/deploy.prototxt'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
image_resize=100
image_num=1
#net.blobs['image'].reshape(image_num,3,image_resize,image_resize)
#print net.blobs['data'].data.shape
#transformer = caffe.io.Transformer({'image': net.blobs['image'].data.shape})
#transformer.set_transpose('image', (2, 0, 1))
#transformer.set_mean('image', np.array([0.5])) # mean pixel
#transformer.set_raw_scale('data', 128)  # the reference model operates on images in [0,255] range instead of [0,1]
#transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


import cv2
from cv2 import cv
import os
import imghdr
import re
import numpy as np
print "ok"

train_image_dir="/home/brl/data/unrolling_v1"
all_files = os.listdir(train_image_dir)
all_files.sort()

if 1==1:
		frame=cv2.imread('/home/brl/data/unrolling/test_img.jpg')
		frame=(frame-127.5)/128
		frame=frame.transpose((2,0,1))
		net.blobs['image'].data[0,:] = frame
		with open('/home/brl/para.txt','rt') as f:
		  for line in f:
		    para=line.split(' ')
		    para=para[0:len(para)-1]
		    para = [float(i) for i in para]
		    net.blobs['fc6_shape'].data[:] = para[0:199]
		    net.blobs['fc6_exp'].data[:] = para[199:228]
        	    net.blobs['fc6_m'].data[:] = para[228:236]

		net.forward()

		isomap = net.forward()['unrolling'][0,:]*255.0
		print isomap.shape
		new_isomap = isomap.transpose((1,2,0)).astype(np.uint8)
		cv2.imwrite('/home/brl/result.jpg',new_isomap)
		#print isomap.shape
		#cv2.imshow("isomap1",new_isomap)
		#cv2.waitKey(0)
		#cv2.imwrite("/home/brl/haha/a.png",new_isomap)

"""
for file in all_files:
    if file.endswith('jpg'):
		#print file
		frame=cv2.imread(train_image_dir+os.path.sep+file,0)
		print frame.shape
		transformed_image = frame
		net.blobs['image'].data[:] = transformed_image
		para_file=train_image_dir+os.path.sep+file.split('.')[0]+'.txt'
		with open(para_file,'rt') as f:
		  for line in f:
		    para=line.split(' ')
		    para=para[0:len(para)-1]
		    para = [float(i) for i in para]
		    net.blobs['p199'].data[:] = para[0:199]
		    net.blobs['p29'].data[:] = para[199:228]
        	    net.blobs['p8'].data[:] = para[228:236]

		net.forward()

		isomap = net.forward()['unrolling'][0,:]*255.0
		print isomap.shape
		new_isomap = isomap.transpose((1,2,0)).astype(np.uint8)
		#print isomap.shape
		cv2.imshow("isomap1",new_isomap)
		cv2.waitKey(0)
		#cv2.imwrite("/home/brl/haha/a.png",new_isomap)
"""	    	
    


