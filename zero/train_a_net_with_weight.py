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

caffe.set_mode_gpu()
solver=caffe.SGDSolver('/home/scw4750/github/unrolling/zero/A_net/solver_with_weight.prototxt')
#solver.restore('zero/a_net/snapshot_with_weight/a_net_iter_280.solverstate')
solver.net.copy_from('/home/scw4750/github/unrolling/zero/A_net/base1_iter_40000.caffemodel')
use_mat=True
if use_mat==True:
    import cv2
    from cv2 import cv
    import h5py
    matfn='/home/scw4750/github/a_net/shuffle_data_2.mat'
    f=h5py.File(matfn)
    img_data=f['new_data']['img']
    para_data=f['new_data']['para']
    data_len=img_data.shape[0]
    cursor=10000
    batch_size=40
    nter=1000000
    for i in range(nter):
	print i
        for batch_num in range(batch_size):
	    #print cursor+batch_num
	    #print img_data[cursor+batch_num,0]
            img=f[img_data[cursor+batch_num,0]][:]
            img=img.transpose()
            img=(img-127.5)/128.0
            solver.net.blobs['data'].data[batch_num,:]=img
            solver.net.blobs['para'].data[batch_num,:]=f[para_data[cursor+batch_num,0]][:].transpose()
            #print f[para_data[cursor+batch_num,0]][:].transpose()
	    #print data_len
	    #cv2.imshow('img',img)
            #cv2.waitKey(0)
        cursor=cursor+batch_num
        if (cursor+batch_num) > data_len:
            cursor=0
        solver.step(1)
        print solver.net.blobs['loss'].data

if use_mat==False:
    crop_image_236_list='/home/brl/data/a_net/train_list.txt'
    f=open(crop_image_236_list,'rt')
    all_lines=f.readlines()
    print len(all_lines)
    import cv2
    from cv2 import cv
    import os
    import imghdr
    import re
    import numpy as np

    niter=20000000
    image_resize=100
    batch_size=40
    cursor=0
    for i in range(1,niter):
        for batch_num in range(batch_size):
            #print all_lines[cursor+batch_num]
            img=cv2.imread(all_lines[cursor+batch_num].split(' ')[0],0)
            processed_img=(img-127.5)/128.0
            txt_length=len(all_lines[cursor+batch_num].split(' ')[1])
            with open(all_lines[cursor+batch_num].split(' ')[1][0:txt_length-1]) as f:
                para=f.readline()
                para=para.split(' ')[0:len(para.split(' '))-1]
                para=[float(i) for i in para]
                assert(len(para)==236)
            solver.net.blobs['data'].data[batch_num,:]=processed_img
            solver.net.blobs['para'].data[batch_num,:]=para;
        cursor=cursor+batch_size
        if(cursor+batch_size>len(all_lines)):
            cursor=0
        solver.step(1)
        print solver.net.blobs['loss'].data
    f.close()       
