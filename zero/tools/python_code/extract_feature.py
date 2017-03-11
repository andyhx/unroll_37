#!/usr/bin/env python
import numpy as np
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

model_weights = '/home/brl/github/unrolling/zero/A_Rnet/R_net.caffemodel'
model_def='/home/brl/github/unrolling/zero/A_Rnet/A_Rnet.prototxt'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
import cv2
from cv2 import cv
import os
import imghdr
import re
import numpy as np

niter=3
image_resize=128
file_loc='zero/feature_lfw.txt'
if os.path.exists(file_loc):
    os.remove(file_loc)
for i in range(0,niter):
    net.forward()
    feature=net.blobs['eltwise_fc1'].data
    with open(file_loc,'at') as f:
        for i in range(256):
            f.write(str(feature[0][i]))
            f.write(' ')
