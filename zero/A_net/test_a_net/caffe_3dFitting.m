
%clear
%caffe
addpath(genpath('/home/brl/github/unrolling/matlab'));
addpath(genpath('/home/brl/github/faceRec/toolbox_graph'));
cnnModel = 'a_rnet_deploy.prototxt';
use_basic_model=false;
use_gpu=true;
if use_basic_model ~=true
  weights = '/home/brl/zero/A_Rnet/snapshot_freeze_r/a_rnet_iter_1400.caffemodel';
else
  weights='base1_iter_40000.caffemodel';
end
net = caffe.Net(cnnModel, weights, 'test');
if use_gpu
  caffe.set_mode_gpu();
  caffe.set_device(0);
else
    caffe.set_mode_cpu();
end

%the list that test for train data
%img_list='/home/brl/data/300w/test_image_and_its_236_list.txt';

%the list test test for new data
img_list='./test_img_list.txt';
fid=fopen(img_list,'r');
img_txt_info=textscan(fid,'%s %s');


test_index=3; %special case 22(open mouse)
%disp img_txt_info{1,1}{test_index,1}

img = imread('/home/brl/zero/test_a_net/test_data/MICC_0026_I_00656.jpg');
%img=imread('/home/brl/test.jpg');
%img=imresize(img,[100 100]);
%img=rgb2gray(img);
figure, imshow(img);
if use_basic_model~=true
    img=img';
end
data = zeros(100,100,1,1);
data = single(data);
data(:,:,:,1) = (single(img)-127.5)/128;
net.blobs('probe').set_data(data);
net.forward_prefilled();
m=net.blobs('fc6_m').get_data();
p_id=net.blobs('fc6_shape').get_data();
p_exp=net.blobs('fc6_exp').get_data();
% fid=fopen('/home/brl/test.txt','rt');
% content=textscan(fid,'%f ');
% content=content{1};
% p_id=content(1:199);
% p_exp=content(200:228);
% p_m=content(229:236);
%%
load('Model_Expression.mat');
load('Model_Shape.mat');
load('meanstd.mat');

m = m' .* para_std(1:8) + para_mean(1:8);
p_id = p_id' .* para_std(15:213) + para_mean(15:213);
p_exp = p_exp' .* para_std(214:242) + para_mean(214:242);
M(1, :) = m(1:4);
M(2, :) = m(5:8);
para=[p_id,p_exp,m];
vertex3d = mu_shape + w * p_id' + mu_exp + w_exp * p_exp';
vertex3d = reshape(vertex3d, 3, length(vertex3d)/3);
vertex4d = vertex3d;
vertex4d(4, :) = 1;
vertex2d = M*vertex4d;
vertex2d(2,:) = size(img, 1) + 1 - vertex2d(2,:);
landmark = vertex2d(:, keypoints);
hold on;
plot(vertex2d(1,:), vertex2d(2,:), 'g.');
%plot(landmark(1,:), landmark(2,:), 'r*');
figure, plot_mesh(vertex3d', tri');

%f=getframe(gca);
%img=frame2im(f);
%imshow(img);
%imwrite(img,'/home/brl/test.jpg')

