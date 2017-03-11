%clear;
addpath(genpath('/home/brl/github/matlab_learning/toolbox_graph'));

%load('data(p&m).mat');
img_list='/home/brl/data/300w/test_image_and_its_236_list.txt';
%a=importdata(img_list);
fid=fopen(img_list,'r');
img_txt_info=textscan(fid,'%s %s');
test_index=22; %special case 22(open mouse)
fid=fopen(img_txt_info{1,2}{test_index,1});
img=imread(img_txt_info{1,1}{test_index,1});

%disp img_txt_info{1,2}{test_index,1}
para=textscan(fid,'%f');
para=para{1,1}';
para_true=para;
fclose(fid);
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
imshow(img),hold on;
plot(vertex2d(1,:), vertex2d(2,:), 'g.');
figure, plot_mesh(vertex3d', tri');

% %% liu
%  P = p .* para_std(1:228) + para_mean(1:228);
%  m = m .* para_std(229:end) + para_mean(229:end);
%  M(1, :) = m(1:4);
%  M(2, :) = m(5:8);
%  M = M + M0;
% 
% % our test
% %P = p;  
% %M(1,:) = m(1:4);
% %M(2,:) = m(5:8);
% vertex3d = mu_shape + w * P(1:199)' + mu_exp + w_exp * P(200:end)';
% vertex3d = reshape(vertex3d, 3, length(vertex3d)/3)';
% figure,plot_mesh(vertex3d',tri');