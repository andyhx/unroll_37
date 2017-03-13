clear;
addpath('/home/scw4750/github/unrolling/zero/Lightencnn/draw_roc');
% load('../draw_roc/ori_img_gallery_feature.mat');
% load('../draw_roc/ori_img_probe_feature.mat');
% load('/home/scw4750/github/unrolling/zero/Lightencnn/draw_roc/softmax_probe_feature.mat');
% load('/home/scw4750/github/unrolling/zero/Lightencnn/draw_roc/softmax_gallery_feature.mat');
% load('../draw_roc/front_gallery_feature.mat');
% load('../draw_roc/front_probe_feature.mat');
% load('../draw_roc/crop_frontal_gallery_feature.mat');
% load('../draw_roc/crop_frontal_probe_feature.mat');
% load('/home/scw4750/github/unrolling/zero/Lightencnn/draw_roc/lightencnn_gallery_feature.mat');
% load('/home/scw4750/github/unrolling/zero/Lightencnn/draw_roc/lightencnn_probe_feature.mat');

count=1;
test_identity={'BRL_0003' 'BRL_0006' 'BRL_0007' 'BRL_0008' 'BRL_0013' 'BRL_0014' 'BRL_0015' 'BRL_0016' ...
    'BRL_0002' 'BRL_0004' 'BRL_0005' 'BRL_0023' ...
    'MICC_0001' 'MICC_0002' 'MICC_0003' 'MICC_0004' 'MICC_0028' 'MICC_0029'};
for i_te=1:length(test_identity)
   test_index(i_te)=get_index_by_name(test_identity{i_te}); 
end
for i_p=1:length(probe_feature)
    if  ~max(test_index==probe_feature(i_p).index) 
        continue;
    end
    i_p
    feat=probe_feature(i_p).fea';
    for i_g=1:length(gallery_feature)
        feat2=gallery_feature(i_g).fea';
        scores(count)=distance.compute_cosine_score(feat, feat2);
        labels(count)=(probe_feature(i_p).index==gallery_feature(i_g).index);
        if labels(count)==1 || rand()<0.01
            count=count+1;
        end
    end
end
roc = evaluation.evaluate('roc', scores, labels);
fprintf('eer:          %f\n', roc.measure);
fprintf('tpr001:       %f\n', roc.extra.tpr001*100);
fprintf('tpr0001:      %f\n', roc.extra.tpr0001*100);
ROC(scores,labels,10); 