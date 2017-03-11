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
caffe.set_mode_gpu()
caffe.set_device(0)

from google.protobuf import text_format
from caffe.proto import caffe_pb2

solver=caffe.SGDSolver('zero/A_Rnet/solver_freeze_r.prototxt')
#solver.restore('zero/A_Rnet/snapshot_freeze_r/a_rnet_iter_1400.solverstate');
#solver.net.copy_from('zero/a_net/base1_iter_40000.caffemodel')
#solver.net.copy_from('/home/brl/github/unrolling/zero/a_net/snapshot_with_weight_v1/a_net_iter_800.caffemodel')
#solver.net.copy_from('zero/A_Rnet/R_net.caffemodel')
solver.net.copy_from('zero/A_Rnet/snapshot_freeze_a/a_rnet_iter_28400.caffemodel')
import cv2
from cv2 import cv
import os
import imghdr
import re
import numpy as np
test_type=4 # 0 for a net ,1 for r net, 2 for unrolling_layer
niter=10000000
image_resize=128
for i in range(niter):
	print i
	solver.step(1)
	print "loss loss loss"
        print solver.net.blobs['loss'].data
	print "loss loss loss"
        if test_type==2:            
            img=solver.net.blobs['unrolling'].data[0,:]
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
            shape=solver.net.blobs['fc6_shape'].data
            print shape.shape
            exp=solver.net.blobs['fc6_exp'].data
            m=solver.net.blobs['fc6_m'].data
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
            img=solver.net.blobs['unrolling'].data[0,:]
            img_pair=solver.net.blobs['data_p'].data[0,:]
            img=img*255.0
            img_pair=img_pair*255.0
            print img.shape
            print img_pair.shape
            cv2.imshow("unrolling",img.transpose((1,2,0)).astype(np.uint8))
            cv2.imshow("img",img_pair.transpose((1,2,0)).astype(np.uint8))
            cv2.waitKey(0)
