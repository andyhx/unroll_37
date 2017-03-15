clear;
load('Model_Expression.mat');
load('Model_Shape.mat');
load('meanstd.mat');
img_list='/home/brl/data/300w/test_image_and_its_236_list.txt';
fid=fopen(img_list,'r');
img_txt_info=textscan(fid,'%s %s');
img_txt_info_size=size(img_txt_info{1,1});
img_num=img_txt_info_size(1);
img_dir='/home/brl/data/300w/crop_image';
for i = 1:20
    img_location= img_txt_info{1,1}{i,1};
    split=strsplit(img_location,'/');
    img_name=split{1,7};
    caffe_3dfitting_fun(img_dir,img_name,true,false);
end
%test_index=22; %special case 22(open mouse)
%disp img_txt_info{1,1}{test_index,1}