import os
file_dir='/home/brl/data/300w/crop_image_v1'
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
