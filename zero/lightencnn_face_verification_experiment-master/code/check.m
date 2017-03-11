weights='/home/brl/zero/A_Rnet/snapshot_freeze_a/a_rnet_iter_2200.caffemodel';
addpath(genpath('/home/brl/github/faceRec/toolbox_graph'));
cnnModel = '/home/brl/github/unrolling/zero/A_Rnet/deploy.prototxt';
net = caffe.Net(cnnModel, weights, 'test');
img_pair = imread('/home/brl/data/a_rnet/img/gallery/BRL_0001.jpg');
img_pair=img_pair';
img = imread('/home/brl/data/a_rnet/img/probe/brl_video/BRL_0001_01_01_000.jpg');
img=img';
data_pair = zeros(128,128,1,1);
data_pair = single(data_pair);
data(:,:,:,1) = (single(img_pair)/255.0);
net.blobs('data_p').set_data(data);

data=zeros(100,100,1,1);
data=single(data);
data(:,:,:,1)=(single(img)-127.5)/128.0;
net.blobs('image').set_data(data);
data_label=zeros(1,1,1,1);
data_label=single(data_label);
data_label(:)=(single(1));
net.blobs('label').set_data(data_label);
net.forward_prefilled();
net.blobs('loss').get_data()

fc1_p=net.blobs('eltwise_fc1_p').get_data();
%data(:,:,:,batch_num) = (single(img)-127.5)/128;