% load data
% load('../results/LightenedCNN_A_lfw.mat');      % model A
% load('../results/LightenedCNN_B_lfw.mat');      % model B
%load('../results/LightenedCNN_C_lfw.mat');      % model C
%load('lfw_pairs.mat');
%clear;
%save trained_probe_feature_iter_1400.mat trained_probe_feature;
load('/home/brl/data/a_rnet/code/train_brl_probe.mat');
load('/home/brl/data/a_rnet/code/train_micc_probe.mat');
load('/home/brl/data/a_rnet/code/pos_pair.mat');
load('/home/brl/data/a_rnet/code/neg_pair.mat');
load('trained_gallery_feature_iter_800.mat');
load('trained_probe_feature_iter_800.mat');

do_basic_evalution=false;
total_feature_num=500;
roc_interval=10;
if do_basic_evalution~=true
        load('trained_gallery_feature_iter_14200.mat');
    load('trained_probe_feature_iter_14200.mat');
    %probe_feature=feature;
    %postive
    % r=randperm(size(same_pair,2));
    % same_pair=same_pair(r);
    % r=randperm(size(diff_pair,2));
    % diff_pair=diff_pair(r);
    count=0;
    for i=1:size(same_pair,2)
        i
        if(same_pair(i).img_pair_index<total_feature_num)
            count=count+1;
            feat1 =  trained_gallery_feature(same_pair(i).img_index).fea';
            feat2 =  trained_probe_feature(same_pair(i).img_pair_index).fea';
            pos_scores(count) = distance.compute_cosine_score(feat1, feat2);
            pdist2(feat1',feat2')
        end
    end
    pos_label = ones(1, count);
    count=0
    for i = 1: size(diff_pair,2)
        i
        if(diff_pair(i).img_pair_index<total_feature_num)
            count=count+1;
            feat1 = trained_gallery_feature(diff_pair(i).img_index).fea';
            feat2 = trained_probe_feature(diff_pair(i).img_pair_index).fea';
            neg_scores(count) = distance.compute_cosine_score(feat1, feat2);
            pdist2(feat1',feat2')
        end
    end
    neg_label = -ones(1, count);
    
    
    scores = [pos_scores, neg_scores];
    label = [pos_label neg_label];
    
    % ap
    ap = evaluation.evaluate('ap', scores, label);
    
    % roc
    roc = evaluation.evaluate('roc', scores, label);
    
    
    %% output
    fprintf('ap:           %f\n', ap.measure);
    fprintf('eer:          %f\n', roc.measure);
    fprintf('tpr001:       %f\n', roc.extra.tpr001*100);
    fprintf('tpr0001:      %f\n', roc.extra.tpr0001*100);
    fprintf('tpr00001:     %f\n', roc.extra.tpr00001*100);
    fprintf('tpr000001:    %f\n', roc.extra.tpr000001*100);
    fprintf('tpr0:         %f\n', roc.extra.tpr0*100);
    result = [ap.measure/100 roc.measure/100  roc.extra.tpr001 roc.extra.tpr0001 roc.extra.tpr00001 roc.extra.tpr000001 roc.extra.tpr0];
    ROC(scores,label,roc_interval,'trained_roc');
    
else
    %probe_feature=feature;
    %postive
    % r=randperm(size(same_pair,2));
    % same_pair=same_pair(r);
    % r=randperm(size(diff_pair,2));
    % diff_pair=diff_pair(r);
    count=0;
    total_feature_num=130000;
    for i=1:size(same_pair,2)
        i
        if(same_pair(i).img_pair_index<total_feature_num)
            count=count+1;
            feat1 =  gallery_feature(same_pair(i).img_index).fea';
            feat2 =  probe_feature(same_pair(i).img_pair_index).fea';
            %pos_scores(count) = distance.compute_cosine_score(feat1, feat2);
            pos_scores(count)=pdist2(feat1,feat2);
            pdist2(feat1',feat2')
        end
    end
    pos_label = ones(1, count);
    count=0
    for i = 1: size(diff_pair,2)
        i
        if(diff_pair(i).img_pair_index<total_feature_num)
            count=count+1;
            feat1 = gallery_feature(diff_pair(i).img_index).fea';
            feat2 = probe_feature(diff_pair(i).img_pair_index).fea';
            %neg_scores(count) = distance.compute_cosine_score(feat1, feat2);
            neg_scores(count)=pdist2(feat1,feat2);
            pdist2(feat1',feat2')
        end
    end
    neg_label = -ones(1, count);
    
    
    scores = [pos_scores, neg_scores];
    label = [pos_label neg_label];
    
    % ap
    ap = evaluation.evaluate('ap', scores, label);
    
    % roc
    roc = evaluation.evaluate('roc', scores, label);
    
    
    %% output
    fprintf('ap:           %f\n', ap.measure);
    fprintf('eer:          %f\n', roc.measure);
    fprintf('tpr001:       %f\n', roc.extra.tpr001*100);
    fprintf('tpr0001:      %f\n', roc.extra.tpr0001*100);
    fprintf('tpr00001:     %f\n', roc.extra.tpr00001*100);
    fprintf('tpr000001:    %f\n', roc.extra.tpr000001*100);
    fprintf('tpr0:         %f\n', roc.extra.tpr0*100);
    result = [ap.measure/100 roc.measure/100  roc.extra.tpr001 roc.extra.tpr0001 roc.extra.tpr00001 roc.extra.tpr000001 roc.extra.tpr0];
    ROC(scores,label,3000,'basic_roc'); 

end
