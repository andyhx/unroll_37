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

basic_dir='/home/scw4750/github/a_rnet/img/probe';
for i = 1:size(probe,1)
    i
    tic
    if strcmp(probe(i).name(1:3),'BRL')
        img = imread([basic_dir filesep 'brl_video' filesep probe(i).name]);
    else
        img=imread([basic_dir filesep 'micc_video' filesep probe(i).name]);
    end
%     img=rgb2gray(img);
    img=img';
%     imshow(img);
    data = zeros(100,100,1,1);
    data = single(data);
    img=(single(img)-127.8)./128.0;
    data(:,:,:,1) = img;
    net.blobs('probe').set_data(data);
    net.forward_prefilled();
    unroll=zeros(128,128,1,1);
%     unroll=net.blobs('unrolling').get_data();
%     figure,imshow(unroll);
    eltwise_fc1=net.blobs('eltwise_fc1').get_data();
    probe_feature(i).name=probe(i).name;
    probe_feature(i).fea=eltwise_fc1';
    toc
end
