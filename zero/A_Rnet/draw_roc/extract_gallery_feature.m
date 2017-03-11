clear;
load('../make_data/gallery.mat');
load('../make_data/probe.mat');

addpath(genpath('/home/scw4750/github/unrolling/matlab'));
% addpath(genpath('/home/brl/github/faceRec/toolbox_graph'));
cnnModel = '/home/scw4750/github/unrolling/zero/A_Rnet/a_rnet_deploy.prototxt';
weights='/home/scw4750/github/unrolling/zero/A_Rnet/snapshot/a_rnet_iter_1.caffemodel';
net=caffe.Net(cnnModel,weights,'test');
use_gpu=true;

if use_gpu
  caffe.set_mode_gpu();
  caffe.set_device(0);
else
    caffe.set_mode_cpu();
end

basic_dir='/home/scw4750/github/a_rnet/img/gallery';
for i = 1:size(gallery,1)
    i
    tic
    img = imread([basic_dir filesep gallery(i).name]);
    %img=rgb2gray(img);
    img=img';
    data = zeros(128,128,1,1);
    data = single(data);
    data(:,:,:,1) = (single(img)/255.0);
    net.blobs('gallery').set_data(data);
    data=zeros(100,100,1,1);
    net.forward_prefilled();
    eltwise_fc1=net.blobs('eltwise_fc1_p').get_data();
    gallery_feature(i).name=gallery(i).name;
    gallery_feature(i).fea=eltwise_fc1';
    toc
end
save gallery_feature.mat gallery_feature;