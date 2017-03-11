clear;

cnnModel = '/home/scw4750/github/unrolling/zero/R_net/deploy.prototxt';
addpath('/home/scw4750/github/unrolling/zero/Lightencnn/tools');
% %for crop_frontal
% weights='/home/scw4750/github/unrolling/zero/Lightencnn/final_LightenedCNN_C.caffemodel';
% probe_dir='/home/scw4750/github/ori&frontal/crop_frontal_img';
% probe_feature=get_feature(probe_dir,cnnModel,weights);
% save crop_frontal_probe_feature.mat probe_feature;
% 
% gallery_dir='/home/scw4750/github/ori&frontal/crop_ori_gallery';
% gallery_feature=get_feature(gallery_dir,cnnModel,weights);
% save crop_frontal_gallery_feature.mat gallery_feature;
% 
% % for our model
% weights='/home/scw4750/github/unrolling/zero/Lightencnn/snapshot/softmax/rnet__iter_376000.caffemodel';
% probe_dir='/home/scw4750/github/r_net/RNpre_probe_img';
% probe_feature=get_feature(probe_dir,cnnModel,weights);
% save softmax_probe_feature.mat probe_feature;
% 
% gallery_dir='/home/scw4750/github/r_net/gallery';
% gallery_feature=get_feature(gallery_dir,cnnModel,weights);
% save softmax_gallery_feature.mat gallery_feature;

%for our img by lightencnn model
weights='/home/scw4750/github/unrolling/zero/Lightencnn/final_LightenedCNN_C.caffemodel';
probe_dir='/home/scw4750/github/r_net/RNpre_probe_img';
probe_feature=get_feature(probe_dir,cnnModel,weights);
save lightencnn_probe_feature.mat probe_feature;
gallery_dir='/home/scw4750/github/r_net/gallery';
gallery_feature=get_feature(gallery_dir,cnnModel,weights);
save lightencnn_gallery_feature.mat gallery_feature;


