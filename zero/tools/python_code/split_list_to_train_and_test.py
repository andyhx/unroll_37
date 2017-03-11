import os
import cv2
crop_image_236_list='/home/brl/data/300w/image_and_its_236_list.txt'
f=open(crop_image_236_list,'rt')
all_lines=f.readlines()
import random
test_file='/home/brl/data/300w/test_image_and_its_236_list.txt'
test_f=open(test_file,'wt')
randnum=[]
for i in range(330):
    randnum.append(random.randint(0,len(all_lines)-1))
print len(randnum)
for num in randnum:
    test_f.write(all_lines[num])
test_f.close()
shuffle_all_line=[]
for i in range(len(all_lines)):
    if i not in randnum:
        shuffle_all_line.append(all_lines[i])
import random
random.shuffle(shuffle_all_line)
new_file='/home/brl/data/300w/train_image_and_its_236_list.txt'
new_f=open(new_file,'wt')
new_f.writelines(shuffle_all_line)
new_f.close()
