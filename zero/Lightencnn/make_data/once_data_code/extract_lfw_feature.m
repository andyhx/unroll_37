addpath('/home/scw4750/github/unrolling/zero/Lightencnn/tools');

root_dir='/home/scw4750/github/LFW_hpen_code/img_LFW';
lfw=dir([root_dir filesep '*.jpg']);

cnnModel = '/home/scw4750/github/unrolling/zero/R_net/deploy.prototxt';

weights='/home/scw4750/github/unrolling/zero/Lightencnn/final_LightenedCNN_C.caffemodel';
feature=extract_feature(root_dir,lfw,cnnModel,weights);

save lfw3d.mat feature

