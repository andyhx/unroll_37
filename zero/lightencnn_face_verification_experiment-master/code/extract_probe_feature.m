
%clear
%caffe
load('/home/brl/data/a_rnet/code/pos_pair.mat');
%load('/home/brl/data/a_rnet/code/neg_pair.mat');

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

%the list test test for new data
%img_list='./test_img_list.txt';
%fid=fopen(img_list,'r');
%img_txt_info=textscan(fid,'%s %s');

%load('same_pair.mat')

%test_index=3; %special case 22(open mouse)
%disp img_txt_info{1,1}{test_index,1}
basic_dir='/home/brl/data/a_rnet/img';
batch=1;
i=0;
total_num=size(same_pair,2);
while i <total_num
    i
    tic
    data = zeros(100,100,1,batch);
    data = single(data);
    for batch_num=1:batch
        img = imread([basic_dir filesep same_pair(i+batch_num).img_pair]);
        %img=imread('/home/brl/test.jpg');
        
        img=img';
        data(:,:,:,batch_num) = (single(img)-127.5)/128;
    end
    net.blobs('probe').set_data(data);
    net.forward_prefilled();
    eltwise_fc1=net.blobs('eltwise_fc1').get_data();
    for batch_num =1:batch
        trained_probe_feature(i+batch_num).name=same_pair(i+batch_num).img_pair;
        trained_probe_feature(i+batch_num).fea=eltwise_fc1(:,batch_num)';
    end
    i=i+batch;
    toc
end
save trained_probe_feature.mat trained_probe_feature;
% m=net.blobs('fc6_m').get_data();
% p_id=net.blobs('fc6_shape').get_data();
% p_exp=net.blobs('fc6_exp').get_data();
% fid=fopen('/home/brl/test.txt','rt');
% content=textscan(fid,'%f ');
% content=content{1};
% p_id=content(1:199);
% p_exp=content(200:228);
% p_m=content(229:236);
%%
% load('Model_Expression.mat');
% load('Model_Shape.mat');
% load('meanstd.mat');
% 
% m = m' .* para_std(1:8) + para_mean(1:8);
% p_id = p_id' .* para_std(15:213) + para_mean(15:213);
% p_exp = p_exp' .* para_std(214:242) + para_mean(214:242);
% M(1, :) = m(1:4);
% M(2, :) = m(5:8);
% para=[p_id,p_exp,m];
% vertex3d = mu_shape + w * p_id' + mu_exp + w_exp * p_exp';
% vertex3d = reshape(vertex3d, 3, length(vertex3d)/3);
% vertex4d = vertex3d;
% vertex4d(4, :) = 1;
% vertex2d = M*vertex4d;
% vertex2d(2,:) = size(img, 1) + 1 - vertex2d(2,:);
% landmark = vertex2d(:, keypoints);
% hold on;
% plot(vertex2d(1,:), vertex2d(2,:), 'g.');
% % plot(landmark(1,:), landmark(2,:), 'r*');
% figure, plot_mesh(vertex3d', tri');

%f=getframe(gca);
%img=frame2im(f);
%imshow(img);
%imwrite(img,'/home/brl/test.jpg')
