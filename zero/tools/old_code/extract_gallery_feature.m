clear;
gallery=dir('/home/brl/data/a_rnet/img/gallery/*.jpg');

addpath(genpath('/home/brl/github/unrolling/matlab'));
addpath(genpath('/home/brl/github/faceRec/toolbox_graph'));
cnnModel = '/home/brl/github/unrolling/zero/A_Rnet/deploy.prototxt';
use_basic_model=true;
use_gpu=true;
if use_basic_model ~=true
  weights = '/home/brl/github/unrolling/zero/A_Rnet/R_net.caffemodel';
else
  weights='/home/brl/github/unrolling/zero/A_Rnet/R_net.caffemodel';
end
net = caffe.Net(cnnModel, weights, 'test');
if use_gpu
  caffe.set_mode_gpu();
  caffe.set_device(0);
else
    caffe.set_mode_cpu();
end
if use_basic_model~=true
    img=img';
end



basic_dir='/home/brl/data/a_rnet/img/gallery';
for i = 1:size(gallery,1)
    i
    img = imread([basic_dir filesep gallery(i).name]);
    %img=imread('/home/brl/test.jpg');
    %figure, imshow(img);
    img=img';
    data = zeros(128,128,1,1);
    data = single(data);
    data(:,:,:,1) = (single(img)-127.5)/128;
    net.blobs('data_p').set_data(data);
    net.forward_prefilled();
    eltwise_fc1=net.blobs('eltwise_fc1_p').get_data();
    gallery_feature(i).name=gallery(i).name;
    gallery_feature(i).fea=eltwise_fc1';
end
for i=1:size(gallery,1)
    
end