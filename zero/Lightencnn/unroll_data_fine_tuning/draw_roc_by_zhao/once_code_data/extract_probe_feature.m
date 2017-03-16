clear;
gallery=dir();
probe=dir();

addpath(genpath('/home/scw4750/github/unrolling/matlab'));
addpath(genpath('/home/brl/github/faceRec/toolbox_graph'));
cnnModel = '/home/scw4750/github/unrolling/zero/R_net/deploy.prototxt';
weights='/home/scw4750/zero/Lightencnn/snapshot/centerloss/rnet__iter_166000.caffemodel';
net=caffe.Net(cnnModel,weights,'test');
use_gpu=true;

if use_gpu
  caffe.set_mode_gpu();
  caffe.set_device(0);
else
    caffe.set_mode_cpu();
end

basic_dir='/home/scw4750/github/r_net/RNpre_probe_img';
for i = 1:size(probe,1)
    i
    tic
    img = imread([basic_dir filesep probe(i).name]);
    img=rgb2gray(img);
    img=img';
    data = zeros(128,128,1,1);
    data = single(data);
    data(:,:,:,1) = (single(img)/255.0);
    net.blobs('image').set_data(data);
    net.forward_prefilled();
    eltwise_fc1=net.blobs('eltwise_fc1').get_data();
    probe_feature(i).name=probe(i).name;
    probe_feature(i).fea=eltwise_fc1';
    toc
end
save lightencnn_from_normal_image_probe_feature.mat probe_feature;