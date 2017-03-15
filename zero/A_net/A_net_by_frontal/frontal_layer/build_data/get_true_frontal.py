#!/usr/bin/env python
import numpy as np
import timeit
from time import clock
import sys,os
caffe_root='/home/scw4750/github/unrolling'
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(1)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

#load net and do preproccessing
model_weights = '/home/scw4750/github/unrolling/zero/A_net/A_net_by_frontal/frontal_layer/frontal.caffemodel'
model_def='/home/scw4750/github/unrolling/zero/A_net/A_net_by_frontal/frontal_layer/deploy_test_backward.prototxt'
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
import h5py
print "ok"

matfn='/home/scw4750/github/a_net/data_(p&m).mat'
f=h5py.File(matfn)
name_data=f['data']['name']
data_len=name_data.shape[0]
for index in range(80000,data_len):
		print index
 	        img_data=f['data']['img']
  		para_data=f['data']['para']
  	        name_data=f['data']['name']
  	        data_len=img_data.shape[0]
		frame=f[img_data[index,0]][:].transpose()
		#print f[name_data[index,0]][:].transpose()[0]
		name=''.join([chr(i) for i in f[name_data[index,0]][:].transpose()[0]])+'.jpg'

		cv2.imwrite("/home/scw4750/github/unrolling/zero/A_net/A_net_by_frontal/frontal_layer/data/test_img"+os.path.sep+name,frame)
		para=f[para_data[index,0]][:].transpose()[0]

		#print name
		frame=(frame-127.5)/128
		#frame=frame.transpose((2,0,1))
		net.blobs['image'].data[0,:] = frame
		net.blobs['pid'].data[0,:] = para[0:199]
		net.blobs['pexp'].data[0,:] = para[199:228]
        	net.blobs['pm'].data[0,:] = para[228:236]

		#net.forward()
		net.forward()	

		isomap =net.blobs['frontal'].data[0,:]*255.0;

		new_isomap = isomap.transpose((1,2,0)).astype(np.uint8)
		#cv2.imwrite('/home/brl/result.jpg',new_isomap)
		#print isomap.shape
		net.backward();
		#cv2.imshow("isomap1",new_isomap)
		#cv2.waitKey(10)
		cv2.imwrite("/home/scw4750/github/unrolling/zero/A_net/A_net_by_frontal/frontal_layer/data/test_true_frontal"+os.path.sep+name,new_isomap)



