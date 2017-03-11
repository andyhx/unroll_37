import os
import cv2
from cv2  import cv
import shutil
crop_image_dir="/home/brl/data/300w/crop_image_v1"
output_dir="/home/brl/data/300w/crop_image"
all_files=os.listdir(crop_image_dir)
for i in range(len(all_files)):
    if all_files[i].endswith('.jpg'):
        print i
        img=cv2.imread(crop_image_dir+os.path.sep+all_files[i])
        img=cv2.cvtColor(img,cv2.cv.CV_BGR2GRAY)
        img=cv2.resize(img,(100,100))
        cv2.imwrite(output_dir+os.path.sep+all_files[i],img)
        txt_name=all_files[i].split('.')[0]+'_para.txt'
        shutil.copyfile(os.path.join(crop_image_dir,txt_name),os.path.join(output_dir,txt_name))
