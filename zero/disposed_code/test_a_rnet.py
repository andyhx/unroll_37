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

model_weights = 'zero/A_Rnet/snapshot/a_rnet_iter_800.caffemodel'
model_def='zero/A_Rnet/deploy.prototxt'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
import cv2
from cv2 import cv
import os
import imghdr
import re
import numpy as np
test_type=4 # 0 for a net ,1 for r net, 2 for unrolling_layer
niter=1
image_resize=128
if 1==1:
	img=cv2.imread('/home/brl/data/a_rnet/img/probe/brl_video/BRL_0001_01_01_030.jpg',0)
	img_pair=cv2.imread('/home/brl/data/a_rnet/img/gallery/BRL_0001.jpg',0)
	net.blobs['image'].data[0,:]=img
	net.blobs['data_p'].data[0,:]=img_pair
	net.blobs['label'].data[0,:]=1
	net.forward()
	isomap=net.blobs['unrolling'].data[0,:]
	isomap=isomap*255
        cv2.imshow("isomap",isomap.transpose((1,2,0)).astype(np.uint8))
        cv2.waitKey(0)
        if test_type==2:
            print net.blobs['loss'].data
            img=net.blobs['unrolling'].data[0,:]
            print img.shape
            #img=img.reshape((128,128))
            img=img*255.0
            img=img.transpose((1,2,0))
            img=img.astype(np.uint8)
            cv2.imshow("img",img)
            cv2.waitKey(0)
        if test_type==0:
            img=solver.net.blobs['image'].data
            img=img.reshape((100,100))
            img=img*128.0+127.5
            img=img.astype(np.uint8)
            cv2.imwrite('/home/brl/test.jpg',img)
            shape=net.blobs['fc6_shape'].data
            print shape.shape
            exp=net.blobs['fc6_exp'].data
            m=net.blobs['fc6_m'].data
            para=[]
            for i in range(shape.shape[1]):
                para.append(shape[0][i])
            for i in range(exp.shape[1]):
                para.append(exp[0][i])
            for i in range(m.shape[1]):
                para.append(m[0][i])
            print  len(para)
            print para
            with open('/home/brl/test.txt','wt') as f:
                for i in range(len(para)):
                    f.write(str(para[i])+' ')
        if test_type==1:
            img=net.blobs['unrolling'].data[0,:]
            img_pair=net.blobs['data_p'].data[0,:]
            img=img*255.0
            img_pair=img_pair*255.0
            print img.shape
            print img_pair.shape
            cv2.imshow("unrolling",img.transpose((1,2,0)).astype(np.uint8))
            cv2.imshow("img",img_pair.transpose((1,2,0)).astype(np.uint8))
            cv2.waitKey(0)
