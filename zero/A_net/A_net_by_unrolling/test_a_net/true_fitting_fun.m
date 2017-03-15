function true_fitting_fun(para_dir,para_name,is_write)
%clear;
load('Model_Expression.mat');
load('Model_Shape.mat');
load('meanstd.mat');

addpath(genpath('/home/brl/github/matlab_learning/toolbox_graph'));

fid=fopen([para_dir '/' para_name]);
%disp img_txt_info{1,2}{test_index,1}
para=textscan(fid,'%f');
para=para{1,1}';
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
close all;
figure, plot_mesh(vertex3d', tri');
if is_write
 f=getframe(gca);
 img=frame2im(f);
%imshow(img);
 imwrite(img,['/home/brl/data/300w/test_result/' para_name(1:end-4) '_true_fitting.jpg'])
end
end
