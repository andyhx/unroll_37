import os
import cv2
from cv2 import cv
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py
matfn='/home/brl/data/a_net/less_v7_3_data.mat'
f=h5py.File(matfn)
data=f['data']['img'][:]
print data
print f.keys()
print type(f['data']['name'])
print f['data']['name'].shape
print f['data']['para'].shape
img=f['data']['img'][0,0]
print img
print type(img)
cv2.imshow("img",img)
cv2.waitKey(0)

#test=f['data'][0][0]
#print test.shape
#print test[0]
#print "1"
#print test[1].shape
#print "2"
#print test[2].shape
#print "3"
#print test[3].shape
#print type(test)
#print test.shape
#img=test[3]
#img=(img-127.5)/128
#print img
