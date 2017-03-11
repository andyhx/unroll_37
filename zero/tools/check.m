clear;
load('img_name_landmark.mat');
addpath(genpath('/home/brl/github/matlab_learning/toolbox_graph'));

i=10030;
for i=1:100
    crop_img_dir='/home/brl/data/300w/crop_image'
    img=imread([crop_img_dir '/' img_name_landmark(i).name]);
    pt2d=img_name_landmark(i).landmark;
    close all;
    figure;
    imshow(img),hold on;
    plot(pt2d(1,:), pt2d(2,:), 'g.');
    hold off;
end
%figure, plot_mesh(vertex3d', tri');