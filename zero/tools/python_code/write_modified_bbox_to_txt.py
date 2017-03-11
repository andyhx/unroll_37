#because we need the new landmarks ,so we should know the origin landmarks and modified bbox;
import cv2
from  cv2 import cv
import os
out_dir="/home/brl/data/300w/crop_image_v1"
all_file_dir=['AFW','AFW_Flip','HELEN','HELEN_Flip','IBUG','IBUG_Flip','LFPW','LFPW_Flip']
#all_file_dir=['MTCNNv1']
basic_dir="/home/brl/data/300w"
#basic_dir="/home/brl/github/MTCNN_face_detection_alignment/code/codes"
count=0
special_count=0
ratio=1.3
for file_dir in all_file_dir:
    all_file=os.listdir(os.path.join(basic_dir,file_dir))
    all_file.sort()
    old_count=count
    for file in all_file:
        if file.endswith('box.txt'):
            count=count+1
            print "the number of writted image"+str(count)
            img_name=file[0:len(file)-8]+".jpg"
            img=cv2.imread(os.path.join(basic_dir,file_dir)+os.path.sep+img_name)
            #print img[450][450][0]
            #print os.path.join(os.path.join(basic_dir,file_dir),file)
            with open(os.path.join(os.path.join(basic_dir,file_dir),file),'rt') as f:
                bbox=f.readlines()[0].split(' ')
                topleft_x=int(float(bbox[0]))
                topleft_y = int(float(bbox[1]))
                downright_x=int(float(bbox[2]))+topleft_x
                downright_y=int(float(bbox[3]))+topleft_y
            height=downright_y-topleft_y
            width=downright_x-topleft_x
            if height>width:
                diff=height-width
                topleft_x=topleft_x-diff/2
                downright_x=downright_x+diff/2
                width=height
            else:
                diff=width-height
                topleft_y=topleft_y-diff/2
                downright_y=downright_y+diff/2
                height=width
            topleft_x=int(topleft_x-width*(ratio -1)/2)
            downright_x=int(downright_x+width*(ratio-1)/2)
            topleft_y=int(topleft_y-height*(ratio-1)/2)
            downright_y=int(downright_y+height*(ratio-1)/2)
            height=downright_y-topleft_y
            width=downright_x-topleft_x
            bbox_name=os.path.join(os.path.join(basic_dir,file_dir),file)
            bbox_name_len=len(bbox_name)
            modified_bbox_name=bbox_name[0:bbox_name_len-7]+'bbox_modified.txt'
            with open(modified_bbox_name,'wt') as f:
                f.write(str(topleft_x)+' ')
                f.write(str(topleft_y)+' ')
                f.write(str(width)+' ')
                f.write(str(height)+' ')
