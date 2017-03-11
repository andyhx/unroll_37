# just copy the codefrom create_crop_image_and_its236_list.py and split_train_test_set.py
import os
basic_dir='/home/brl/data/300w'
file_dir='crop_image'
output_dir='/home/brl/data/300w'
all_file=os.listdir(file_dir)
all_file.sort()
output_file='image_and_its_236_list.txt'
with open(os.path.join(output_dir,output_file),'wt') as f:
    for file in all_file:
        if file.endswith('.jpg'):
            txt_name=file[0:len(file)-4]+'_para.txt'
            assert(os.path.exists(os.path.join(file_dir,txt_name)))
            f.write(os.path.join(file_dir,file))
            f.write(' ')
            f.write(os.path.join(file_dir,txt_name))
            f.write('\n')

crop_image_236_list=output_dir+os.path.sep+output_file
f=open(crop_image_236_list,'rt')
all_lines=f.readlines()
import random
test_file=basic_dir+os.path.sep+'test_image_and_its_236_list.txt'
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
new_file=basic_dir+os.path.sep+'train_image_and_its_236_list.txt'
new_f=open(new_file,'wt')
new_f.writelines(shuffle_all_line)
new_f.close()
