clear;

cnnModel = '/home/scw4750/github/unrolling/zero/R_net/deploy.prototxt';

addpath('/home/scw4750/github/unrolling/zero/Lightencnn/tools');
%for crop_frontal
weights='/home/scw4750/github/unrolling/zero/Lightencnn/final_LightenedCNN_C.caffemodel';
basic_dir='/home/scw4750/github/ori&frontal/crop_ori_gallery'

gallery_feature=get_feature(basic_dir,cnnModel,weights);
save crop_frontal_gallery_feature.mat gallery_feature


% function result=get_feature(basic_dir,cnnModel,weights)
% 
% feature=dir([basic_dir filesep '*.jpg']);
% feature=write_index(feature);
% result=extract_feature(basic_dir,feature,cnnModel,weights);
% 
% end
% % for unrolling img
% % weights='/home/scw4750/github/unrolling/zero/Lightencnn/snapshot/softmax/rnet__iter_376000.caffemodel';
% % basic_dir='/home/scw4750/github/r_net/gallery';
% % 
% % gallery=dir('/home/scw4750/github/r_net/gallery/*.jpg');
% % gallery=write_index(gallery);
% % 
% % gallery_feature=extract_feature('/home/scw4750/github/r_net/gallery',gallery,cnnModel,weights);
% % save unrolling_gallery_feature.mat gallery_feature
% 
% 
% 
% function result=write_index(img)
% 
% for i_p=1:length(img)
%    img(i_p).index=get_index_by_name(img(i_p).name);
% end
% result=img;
% end
