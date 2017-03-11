
#!/usr/bin/env python
import numpy as np
import timeit
from time import clock
import sys,os
#find the directory of the script

# Make sure that caffe is on the python path:
caffe_root = '/home/scw4750/github/unrolling' # this file is expected to be in {caffe_root}/examples
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
model='zero/A_net/unrolling_layer/deploy.prototxt'
weight='zero/A_net/unrolling_layer/unrolling.caffemodel'
net=caffe.Net(model,weight,caffe.TEST)

use_mat=True
if use_mat==True:
    import cv2
    from cv2 import cv
    import h5py
    basic_dir='/home/scw4750/zero/A_net/unrolling_layer/test_img/lfw/lfw'
    matfn='/home/scw4750/github/unrolling/zero/A_net/unrolling_layer/test_img/lfw/lfw3D.mat'
    f=h5py.File(matfn)
    img_data=f['lfw3d']['imgPath']
    para_data=f['lfw3d']['para']
    data_len=img_data.shape[0]
    #print img_data[cursor+batch_num,0]
    for index in range(data_len):
	    print index
	    img_path=[chr(i) for i in f[img_data[index,0]][:]]
   	    person_name=''.join(img_path).split('\\')[0]
	    img_name=''.join(img_path).split('\\')[1]
	    path=basic_dir+os.path.sep+person_name+os.path.sep+img_name
	    img=cv2.imread(path,0)
	    img=(img-127.5)/128.0
	    net.blobs['image'].data[0,:]=img
	    para=f[para_data[index,0]][:].transpose()[0]
	    net.blobs['p199'].data[:]=para[0:199]
	    net.blobs['p29'].data[:]=para[199:228]
	    net.blobs['p8'].data[:]=para[228:236]
	    net.forward()
	    unrolling=net.blobs['unrolling'].data[0,0,:]
	    cv2.imwrite('/home/scw4750/github/unrolling/zero/A_net/unrolling_layer/true_isomap/LFW/'+img_name+'.jpg',(unrolling*255).astype(np.uint8))


