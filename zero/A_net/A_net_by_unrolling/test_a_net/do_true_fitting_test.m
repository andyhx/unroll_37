
clear;

% img_list='/home/brl/data/300w/test_image_and_its_236_list.txt';
% fid=fopen(img_list,'r');
% img_txt_info=textscan(fid,'%s %s');
% img_txt_info_size=size(img_txt_info{1,1});
% para_num=img_txt_info_size(1);
% para_dir='/home/brl/data/300w/crop_image';
% for i = 1:20
%     para_location= img_txt_info{1,2}{i,1};
%     split=strsplit(para_location,'/');
%     para_name=split{1,7};
%     true_fitting_fun(para_dir,para_name,true);
% end


addpath(genpath('/home/brl/github/matlab_learning/toolbox_graph'));


% %fid=fopen([para_dir '/' para_name]);
% %disp img_txt_info{1,2}{test_index,1}
load('data_3(p&m).mat');
load('Model_Expression.mat');
load('Model_Shape.mat');
load('meanstd.mat');
%for i = 1:10
for i=1:80
img=imread(['/home/brl/data/300w/crop_image' filesep data(i).name '.jpg']);
%imwrite(img,['/home/brl' filesep data(i).name '.jpg']);
%end 
figure,imshow(img);
para=data(i).para;
%para=para{1,1}';
%fclose(fid);
p_id = para(1:199);
p_exp = para(200:228);
m=para(229:236);
m = m .* para_std(1:8) + para_mean(1:8);
p_id = p_id .* para_std(15:213) + para_mean(15:213);
p_exp = p_exp .* para_std(214:242) + para_mean(214:242);
M(1, :) = m(1:4);
M(2, :) = m(5:8);

vertex3d = mu_shape + w * p_id' + mu_exp + w_exp * p_exp';
vertex3d = reshape(vertex3d, 3, length(vertex3d)/3);
vertex4d = vertex3d;
vertex4d(4, :) = 1;
vertex2d = M*vertex4d;
vertex2d(2,:) = size(img, 1) + 1 - vertex2d(2,:);
figure;
%imshow(img),hold on;
%plot(vertex2d(1,:), vertex2d(2,:), 'g.');
%figure, plot_mesh(vertex3d', tri');
end