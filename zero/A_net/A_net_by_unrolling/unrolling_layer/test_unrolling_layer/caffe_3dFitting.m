
%clear
%caffe
%load('/home/scw4750/github/a_net/shuffle_data.mat');
index=6667;
img = new_data(index).img;
figure,imshow(img);
figure,imshow(img);
para=new_data(index).para;
p_id=para(1:199)';
p_exp=para(200:228)';
m=para(229:236)';
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
%figure, plot_mesh(vertex3d', tri');

%f=getframe(gca);
%img=frame2im(f);
%imshow(img);
%imwrite(img,'/home/brl/test.jpg')

