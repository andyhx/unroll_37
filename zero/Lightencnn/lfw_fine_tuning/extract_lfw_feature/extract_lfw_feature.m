addpath('/home/scw4750/github/unrolling/zero/Lightencnn/tools');
clear
% root_dir='/home/scw4750/github/LFW_hpen_code/img_LFW';
% root_dir='/home/scw4750/github/lfw3d';
% root_dir='/home/scw4750/github/unrolling/zero/A_net/unrolling_layer/true_isomap/LFW';
% 
% 
% cnnModel = '/home/scw4750/github/unrolling/zero/R_net/deploy.prototxt';
% 
% weights='/home/scw4750/github/unrolling/zero/Lightencnn/final_LightenedCNN_C.caffemodel';
% weights='/home/scw4750/github/unrolling/zero/Lightencnn/snapshot/frontal/rnet__iter_20000.caffemodel';
% weights='/home/scw4750/github/unrolling/zero/Lightencnn/snapshot/zhao_softmax_with_low_interval_caffemodel/rnet__iter_8000.caffemodel';
% feature=extract_feature(root_dir,lfw,cnnModel,weights);
% 
% save hpen_tuning.mat feature

cnnModel = '/home/scw4750/github/unrolling/zero/R_net/deploy.prototxt';

% weights='/home/scw4750/github/unrolling/zero/Lightencnn/final_LightenedCNN_C.caffemodel';
% root_dir='/home/scw4750/github/unrolling/zero/A_net/unrolling_layer/true_isomap/LFW';
% extract_lfw_feature_v1(root_dir,cnnModel,weights,'new_our.mat');
% root_dir='/home/scw4750/github/LFW_hpen_code/img_LFW';
% extract_lfw_feature_v1(root_dir,cnnModel,weights,'new_hpen.mat');
% root_dir='/home/scw4750/github/lfw3d';
% extract_lfw_feature_v1(root_dir,cnnModel,weights,'new_lfw3d.mat');

% weights='/home/scw4750/github/unrolling/zero/Lightencnn/lfw_fine_tuning/snapshot/lfw_our__iter_10000.caffemodel';
% root_dir='/home/scw4750/github/unrolling/zero/A_net/unrolling_layer/true_isomap/LFW';
% extract_lfw_feature_v1(root_dir,cnnModel,weights,'new_our_tuning.mat');

weights='/home/scw4750/github/unrolling/zero/Lightencnn/lfw_fine_tuning/snapshot/lfw_zhu__iter_13800.caffemodel'
root_dir='/home/scw4750/github/LFW_hpen_code/img_LFW';
extract_lfw_feature_v1(root_dir,cnnModel,weights,'new_hpen_tuning.mat');
% root_dir='/home/scw4750/github/lfw3d';
% extract_lfw_feature_v1(root_dir,cnnModel,weights,'new_lfw3d_tuning.mat');

function extract_lfw_feature_v1(root_dir,cnnModel,weights,out_mat)
  lfw=dir([root_dir filesep '*.jpg']);
  feature=extract_feature(root_dir,lfw,cnnModel,weights);
  save(out_mat,'feature');
end
