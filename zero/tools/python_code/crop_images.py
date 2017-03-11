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
            if topleft_x>=0 and topleft_x<img.shape[1] and topleft_y>=0 and topleft_y<img.shape[0] \
            and downright_x>=0  and downright_x<img.shape[1] and downright_y>=0 and downright_y<img.shape[0]:
                crop_img=img[topleft_y:downright_y,topleft_x:downright_x]
            else:
                special_count=special_count+1
                print "special count:"+str(special_count)
                with open('/home/brl/data/300w/special_crop_image_name.txt','at') as f:
                    f.write(img_name+'\n')
                import numpy as np
                crop_img=np.zeros(shape=(height,width,img.shape[2]),dtype=np.uint8)
                for i in range(height):
                    for j in range(width):
                        for c in range(img.shape[2]):
                            if topleft_y+i>=0 and topleft_y+i<img.shape[0] and topleft_x+j>=0 and topleft_x+j<img.shape[1]:
                                crop_img[i][j][c]=img[topleft_y+i][topleft_x+j][c]  
            cv2.imwrite(out_dir+os.path.sep+img_name,crop_img)
