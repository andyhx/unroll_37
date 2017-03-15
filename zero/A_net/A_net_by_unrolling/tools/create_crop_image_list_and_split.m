basic_dir='/home/brl/data/300w';
file_dir='crop_image';
output_dir='/home/brl/data/300w';
output_file='image_and_its_236_list_v1.txt';

all_file=dir([basic_dir filesep file_dir filesep '*.jpg']);
all_file_size=size(all_file);

fid=fopen([output_dir filesep output_file],'wt');
for i =1:all_file_size(1)
img_name=all_file(i).name;
para_name=[img_name(1:end-4) '_para.txt'];
fprintf(fid,'%s %s\n',[basic_dir filesep file_dir filesep ...
    img_name],[basic_dir filesep file_dir filesep ...
    para_name]);
end
fclose(fid);

% crop_image_236_list=output_dir+os.path.sep+output_file
% f=open(crop_image_236_list,'rt')
% all_lines=f.readlines()
% import random
% test_file=basic_dir+os.path.sep+'test_image_and_its_236_list.txt'
% test_f=open(test_file,'wt')
% randnum=[]
% for i in range(330):
%     randnum.append(random.randint(0,len(all_lines)-1))
% print len(randnum)
% for num in randnum:
%     test_f.write(all_lines[num])
% test_f.close()
% shuffle_all_line=[]
% for i in range(len(all_lines)):
%     if i not in randnum:
%         shuffle_all_line.append(all_lines[i])
% import random
% random.shuffle(shuffle_all_line)
% new_file=basic_dir+os.path.sep+'train_image_and_its_236_list.txt'
% new_f=open(new_file,'wt')
% new_f.writelines(shuffle_all_line)
% new_f.close()
