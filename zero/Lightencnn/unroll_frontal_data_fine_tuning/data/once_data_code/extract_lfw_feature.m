addpath('/home/scw4750/github/unrolling/zero/Lightencnn/tools');
clear
root_dir='/home/scw4750/github/LFW_hpen_code/img_LFW';
% root_dir='/home/scw4750/github/lfw3d';
% root_dir='/home/scw4750/github/unrolling/zero/A_net/unrolling_layer/true_isomap/LFW';
lfw=dir([root_dir filesep '*.jpg']);

cnnModel = '/home/scw4750/github/unrolling/zero/R_net/deploy.prototxt';
% 
% weights='/home/scw4750/github/unrolling/zero/Lightencnn/final_LightenedCNN_C.caffemodel';
weights='/home/scw4750/github/unrolling/zero/Lightencnn/snapshot/frontal/rnet__iter_20000.caffemodel';
% weights='/home/scw4750/github/unrolling/zero/Lightencnn/snapshot/zhao_softmax_with_low_interval_caffemodel/rnet__iter_8000.caffemodel';
feature=extract_feature(root_dir,lfw,cnnModel,weights);

save hpen_tuning.mat feature

