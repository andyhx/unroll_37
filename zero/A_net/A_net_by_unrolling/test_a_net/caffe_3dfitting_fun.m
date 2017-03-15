function caffe_3dfitting_fun(img_dir,img_name,is_write,use_basic_model)
%clear
%caffe
addpath(genpath('/home/brl/github/unrolling/matlab'));
addpath(genpath('/home/brl/github/faceRec/toolbox_graph'));
cnnModel = 'deploy.prototxt';

if use_basic_model
 weights='base1_iter_40000.caffemodel'
else
 weights = '/home/brl/github/unrolling/zero/a_net/snapshot_v3/a_net_iter_23000.caffemodel';   
end
net = caffe.Net(cnnModel, weights, 'test');
%caffe
%img
%img_list='/home/brl/data/300w/test_image_and_its_236_list.txt';
%fid=fopen(img_list,'r');
%img_txt_info=textscan(fid,'%s %s');
%test_index=22; %special case 22(open mouse)
%disp img_txt_info{1,1}{test_index,1}
img = imread([img_dir '/' img_name]);
%imshow(img)
%img=rgb2gray(img);
%img=imresize(img,[100,100],'bilinear','AntiAliasing',false);
%ung
%img=img';
data = zeros(100,100,1,1);
data = single(data);
data(:,:,:,1) = (single(img)-127.5)/128;
net.blobs('data').set_data(data);
net.forward_prefilled();
m=net.blobs('fc6_m').get_data();
p_id=net.blobs('fc6_shape').get_data();
p_exp=net.blobs('fc6_exp').get_data();
%para_caffe = [p_id;p_exp;m]';

%%
load('Model_Expression.mat');
load('Model_Shape.mat');
load('meanstd.mat');

m = m' .* para_std(1:8) + para_mean(1:8);
p_id = p_id' .* para_std(15:213) + para_mean(15:213);
p_exp = p_exp' .* para_std(214:242) + para_mean(214:242);
M(1, :) = m(1:4);
M(2, :) = m(5:8);

clear vertex3d;
vertex3d = mu_shape + w * p_id' + mu_exp + w_exp * p_exp';
vertex3d = reshape(vertex3d, 3, length(vertex3d)/3);
% vertex4d = vertex3d;
% vertex4d(4, :) = 1;
% vertex2d = M*vertex4d;
% vertex2d(2,:) = size(img, 1) + 1 - vertex2d(2,:);
% landmark = vertex2d(:, keypoints);

%figure, imshow(img); hold on;
%plot(vertex2d(1,:), vertex2d(2,:), 'g.');
% plot(landmark(1,:), landmark(2,:), 'r*');
close all;
figure,plot_mesh(vertex3d', tri');
if is_write
 f=getframe(gca);
 img=frame2im(f);
%imshow(img);
 if use_basic_model
   imwrite(img,['/home/brl/data/300w/test_result/' img_name(1:end-4) '_caffe_basic_fitting.jpg'])
 else
   imwrite(img,['/home/brl/data/300w/test_result/' img_name(1:end-4) '_caffe_fitting.jpg'])  
 end
end
end

