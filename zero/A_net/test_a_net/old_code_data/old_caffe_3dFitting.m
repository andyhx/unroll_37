%clear
addpath(genpath('/home/brl/github/unrolling/matlab'));
addpath(genpath('/home/brl/github/faceRec/toolbox_graph'));
cnnModel = 'deploy.prototxt';
weights = '../a_net/snapshot/a_net__iter_200.caffemodel';
img_list='/home/brl/data/300w/test_image_and_its_236_list.txt';
%a=importdata(img_list);
fid=fopen(img_list,'r');
img_txt_info=textscan(fid,'%s %s');

net = caffe.Net(cnnModel, weights, 'test');
test_index=105;
img = imread(img_txt_info{1,1}{test_index,1});
img=rgb2gray(img);
img=imresize(img,[120,120],'bilinear','AntiAliasing',false);
%imshow(img);
data = zeros(120,120,1,1);
data = single(data);
data(:,:,:,1) = (single(img)-127.5)/128;
net.blobs('data').set_data(data);
net.forward_prefilled();
m=net.blobs('fc6_m').get_data();
pid=net.blobs('fc6_pid').get_data();
pexp=net.blobs('fc6_pexp').get_data();
para = [pid;pexp;m]'

%%
load('Model_Expression.mat');
load('Model_Shape.mat');
load('paras.mat');
p = para(1:228);
m = para(229:end);

P = p .* para_std(1:228) + para_mean(1:228);
m = m .* para_std(229:end) + para_mean(229:end);

M(1, :) = m(1:4);
M(2, :) = m(5:8);
M = M + M0;

vertex3d = mu_shape + w * P(1:199)' + mu_exp + w_exp * P(200:end)';
vertex3d = reshape(vertex3d, 3, length(vertex3d)/3);
%vertex4d = vertex3d;
%vertex4d(4, :) = 1;
%vertex2d = M*vertex4d;
%vertex2d(2,:) = size(img, 1) + 1 - vertex2d(2,:);
%landmark = vertex2d(:, keypoints);

%figure, imshow(img); hold on;
%plot(vertex2d(1,:), vertex2d(2,:), 'g.');
%plot(landmark(1,:), landmark(2,:), 'r*');
figure, plot_mesh(vertex3d', tri');

fclose(fid);
