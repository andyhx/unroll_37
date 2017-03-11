clear;

cnnModel = '/home/scw4750/github/unrolling/zero/R_net/deploy.prototxt';

%for crop_frontal
weights='/home/scw4750/github/unrolling/zero/Lightencnn/final_LightenedCNN_C.caffemodel';
basic_dir='/home/scw4750/github/ori&frontal/crop_frontal_img'
probe_feature=get_feature(basic_dir,cnnModel,weights);
save crop_frontal_probe_feature.mat probe_feature


% function result=get_feature(basic_dir,cnnModel,weights)
% 
% feature=dir([basic_dir filesep '*.jpg']);
% feature=write_index(feature);
% result=extract_feature(basic_dir,feature,cnnModel,weights);
% 
% end
% %%%for origin image
% % probe=dir('/home/scw4750/github/ori&frontal/ori_img/*/*.jpg');
% % for i_p=1:length(probe)
% %   probe(i_p).index=get_index_by_name(probe(i_p).name);
% %   pro_name=probe(i_p).name;
% %   idx=strfind(pro_name,'_');
% %   probe(i_p).name=[pro_name(1:idx-1) filesep pro_name];
% % end
% % probe_feature=extract_feature('/home/scw4750/github/ori&frontal/ori_img',probe,cnnModel,weights);
% % save ori_img_probe_feature.mat probe_feature
% 
% 
% % %%for frontalization img
% % basic_dir='/home/scw4750/github/ori&frontal/frontal_img'
% % probe=dir([basic_dir filesep '*/*.jpg']);
% % probe=write_index(probe);
% % probe_feature=extract_feature(basic_dir,probe,cnnModel,weights);
% % save front_probe_feature.mat probe_feature
% 
% % for unrolling img
% % weights='/home/scw4750/github/unrolling/zero/Lightencnn/snapshot/softmax/rnet__iter_376000.caffemodel';
% % basic_dir='/home/scw4750/github/r_net/RNpre_probe_img'
% % 
% % probe=dir('/home/scw4750/github/r_net/RNpre_probe_img/*.jpg');
% % probe=write_index(probe);
% % 
% % probe_feature=extract_feature('/home/scw4750/github/r_net/RNpre_probe_img',probe,cnnModel,weights);
% % save unrolling_probe_feature.mat probe_feature
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
% 
