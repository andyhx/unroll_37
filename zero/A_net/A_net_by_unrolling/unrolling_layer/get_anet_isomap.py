#!/usr/bin/env python
import numpy as np
import timeit
from time import clock
import sys,os
#find the directory of the script

# Make sure that caffe is on the python path:
caffe_root = '../..' # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

caffe.set_mode_gpu()
model='zero/unrolling_layer/anet_deploy.prototxt'
weight='zero/a_net/snapshot_with_weight_v1/a_net_iter_3400.caffemodel'
net=caffe.Net(model,weight,caffe.TEST)
use_mat=True
if use_mat==True:
    import cv2
    from cv2 import cv
    import h5py
    matfn='../a_net/shuffle_data.mat'
    f=h5py.File(matfn)
    img_data=f['new_data']['img']
    data_len=img_data.shape[0]
    index=6666
    #print img_data[cursor+batch_num,0]
    img=f[img_data[index,0]][:]
    img=img.transpose()
    cv2.imshow("img",img)
    cv2.waitKey(10)
    img=(img-127.5)/128.0
    net.blobs['image'].data[0,:]=img
    net.forward()
    unrolling=net.blobs['unrolling'].data[0,0,:]
    cv2.imwrite('zero/unrolling_layer/anet_isomap/'+str(index)+'.jpg',(unrolling*255).astype(np.uint8))
    cv2.imshow("unroll",unrolling)
    cv2.waitKey(10)
    

