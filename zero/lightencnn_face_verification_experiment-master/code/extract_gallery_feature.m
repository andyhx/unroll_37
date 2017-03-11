%clear;
%gallery=dir('/home/brl/data/a_rnet/img/gallery/*.jpg');
load('/home/brl/data/a_rnet/code/train_gallery.mat');
gallery=train_gallery;


addpath(genpath('/home/brl/github/unrolling/matlab'));
addpath(genpath('/home/brl/github/faceRec/toolbox_graph'));
cnnModel = '/home/brl/github/unrolling/zero/A_Rnet/deploy.prototxt';
use_basic_model=false;
use_gpu=true;

if use_basic_model ==true
  weights='/home/brl/github/unrolling/zero/A_Rnet/R_net.caffemodel';
  net = caffe.Net(cnnModel, weights, 'test');
  net.copy_from('/home/brl/github/unrolling/zero/a_net/snapshot_v6/a_net_iter_54000.caffemodel');
  net.copy_from('/home/brl/github/unrolling/zero/A_Rnet/R_net.caffemodel');
else
  weights='/home/brl/zero/A_Rnet/snapshot_freeze_a/a_rnet_iter_1400.caffemodel';
  net = caffe.Net(cnnModel, weights, 'test');
end

if use_gpu
  caffe.set_mode_gpu();
  caffe.set_device(0);
else
    caffe.set_mode_cpu();
end

basic_dir='/home/brl/data/a_rnet/img/gallery';
for i = 1:size(gallery,1)
    i
    tic
    img = imread([basic_dir filesep gallery(i).name]);
    %img=imread('/home/brl/test.jpg');
    %figure, imshow(img);
    img=img';
    data = zeros(128,128,1,1);
    data = single(data);
    data(:,:,:,1) = (single(img)/255.0);
    net.blobs('gallery').set_data(data);
    net.forward_prefilled();
    eltwise_fc1=net.blobs('eltwise_fc1_p').get_data();
    trained_gallery_feature(i).name=gallery(i).name;
    trained_gallery_feature(i).fea=eltwise_fc1';
    toc
end
save trained_gallery_feature.mat trained_gallery_feature;