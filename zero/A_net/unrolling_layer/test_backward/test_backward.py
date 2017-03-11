#!/usr/bin/env python
import numpy as np
import timeit
from time import clock
import sys,os
#find the directory of the script
# Make sure that caffe is on the python path:
caffe_root = '../..'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_mode_gpu()
caffe.set_device(0)

from google.protobuf import text_format
from caffe.proto import caffe_pb2

model_weights ='/home/scw4750/github/unrolling/zero/test_backward/snapshot/anet_with_unrolling_0.0000000005_iter_600.caffemodel'
model_def='zero/test_backward/deploy.prototxt'
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
niter=10000000
#image_resize=128
all_img=os.listdir('zero/test_backward/test_image')
all_img.sort()
for img in all_img:
    img=cv2.imread('zero/test_backward/test_image/'+img,0)
    img=(img-127.5)/128.0
    net.blobs['probe'].data[:]=img
    net.forward()
