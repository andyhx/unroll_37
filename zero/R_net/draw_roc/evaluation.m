
% load('../make_data/gallery.mat');
% load('../make_data/probe.mat');
clear;
load('gallery_feature.mat');
load('probe_feature.mat');
do_val=true;
if do_val
    load('../make_data/val_pos_pair.mat');
    load('../make_data/val_neg_pair.mat');
    pos_pair=val_pos_pair;
    neg_pair=val_neg_pair;
else
    load('../make_data/pos_pair.mat');
    load('../make_data/neg_pair.mat');
end
do_basic_evalution=false;
total_feature_num=30000;
roc_interval=10;
if do_basic_evalution~=true
    count=0;
    for i=1:size(pos_pair,2)
        i
        if(pos_pair(i).img_pair_index<total_feature_num)
            count=count+1;
            feat1 =  gallery_feature(pos_pair(i).img_index).fea';
            feat2 =  probe_feature(pos_pair(i).img_pair_index).fea';
%             pos_scores(count) = distance.compute_cosine_score(feat1, feat2);
            pos_scores(count) = pdist2(feat1', feat2');
            pos_scores(count)
        end
    end
    pos_label = ones(1, count);
    count=0;
    for i = 1: size(neg_pair,2)
        i
        if(neg_pair(i).img_pair_index<total_feature_num)
            count=count+1;
            feat1 = gallery_feature(neg_pair(i).img_index).fea';
            feat2 = probe_feature(neg_pair(i).img_pair_index).fea';
%             neg_scores(count) = distance.compute_cosine_score(feat1, feat2);
            neg_scores(count) = pdist2(feat1', feat2');
            neg_scores(count)
        end
    end
    neg_label = -ones(1, count);
    
    
    scores = [pos_scores, neg_scores];
    label = [pos_label neg_label];
    
    % ap
%     ap = evaluation.evaluate('ap', scores, label);
    
    % roc
%     roc = evaluation.evaluate('roc', scores, label);
    
    
    %% output
%     fprintf('ap:           %f\n', ap.measure);
%     fprintf('eer:          %f\n', roc.measure);
%     fprintf('tpr001:       %f\n', roc.extra.tpr001*100);
%     fprintf('tpr0001:      %f\n', roc.extra.tpr0001*100);
%     fprintf('tpr00001:     %f\n', roc.extra.tpr00001*100);
%     fprintf('tpr000001:    %f\n', roc.extra.tpr000001*100);
%     fprintf('tpr0:         %f\n', roc.extra.tpr0*100);
%     result = [ap.measure/100 roc.measure/100  roc.extra.tpr001 roc.extra.tpr0001 roc.extra.tpr00001 roc.extra.tpr000001 roc.extra.tpr0];
    ROC_old(scores,label,roc_interval,'trained_roc');
    
    
else
    %probe_feature=feature;
    %postive
    % r=randperm(size(pos_pair,2));
    % pos_pair=pos_pair(r);
    % r=randperm(size(neg_pair,2));
    % neg_pair=neg_pair(r);
    count=0;
    total_feature_num=130000;
    for i=1:size(pos_pair,2)
        i
        if(pos_pair(i).img_pair_index<total_feature_num)
            count=count+1;
            feat1 =  gallery_feature(pos_pair(i).img_index).fea';
            feat2 =  probe_feature(pos_pair(i).img_pair_index).fea';
            %pos_scores(count) = distance.compute_cosine_score(feat1, feat2);
            pos_scores(count)=pdist2(feat1,feat2);
            pdist2(feat1',feat2')
        end
    end
    pos_label = ones(1, count);
    count=0
    for i = 1: size(neg_pair,2)
        i
        if(neg_pair(i).img_pair_index<total_feature_num)
            count=count+1;
            feat1 = gallery_feature(neg_pair(i).img_index).fea';
            feat2 = probe_feature(neg_pair(i).img_pair_index).fea';
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
