% clear;
% 
% load('/home/scw4750/github/unrolling/zero/Lightencnn/draw_roc/unrolling_probe_feature.mat');
% load('/home/scw4750/github/unrolling/zero/Lightencnn/draw_roc/unrolling_gallery_feature.mat');
% load('../draw_roc/front_gallery_feature.mat');
% load('../draw_roc/front_probe_feature.mat');
% load('/home/scw4750/github/unrolling/zero/Lightencnn/draw_roc/unrolling_probe_feature.mat');
% load('/home/scw4750/github/unrolling/zero/Lightencnn/draw_roc/unrolling_gallery_feature.mat');
addpath('/home/scw4750/github/unrolling/zero/Lightencnn/draw_roc');
test_identity={'BRL_0003' 'BRL_0006' 'BRL_0007' 'BRL_0008' 'BRL_0013' 'BRL_0014' 'BRL_0015' 'BRL_0016' ...
    'BRL_0002' 'BRL_0004' 'BRL_0005' 'BRL_0023' ...
    'MICC_0001' 'MICC_0002' 'MICC_0003' 'MICC_0004' 'MICC_0028' 'MICC_0029'};
for i_te=1:length(test_identity)
   test_index(i_te)=get_index_by_name(test_identity{i_te}); 
end

g_feat = zeros(256, length(gallery_feature));
for i = 1:length(gallery_feature)
    g_feat(:, i) = gallery_feature(i).fea;
end

rank_one_true = 0;
test_num=0;
for i = 1:length(probe_feature)
    i
    if ~max(test_index==probe_feature(i).index)
        continue;
    end
    
    test_num=test_num+1;
    pro_name=probe_feature(i).name;
    feat=probe_feature(i).fea(:);
    
    %p_feat = repmat(feat, length(gallery_feature), 1);
    scores = zeros(length(gallery_feature), 1);
    for j = 1:length(gallery_feature)
        scores(j) = distance.compute_cosine_score(g_feat(:, j), feat);
    end
    
    [maxscore, index]=max(scores);
    
    idx = strfind(pro_name, '_');
    if strcmp(gallery_feature(index).name(1:end-4), pro_name(1:idx(2)-1)) == 1
        rank_one_true=rank_one_true+1;
    end
end

rank_one_true/test_num

return;


% test_identity={'BRL_0003' 'BRL_0006' 'BRL_0007' 'BRL_0008' 'BRL_0013' 'BRL_0014' 'BRL_0015' 'BRL_0016' ...
%     'BRL_0002' 'BRL_0004' 'BRL_0005' 'BRL_0023' ...
%     'MICC_0001' 'MICC_0002' 'MICC_0003' 'MICC_0004' 'MICC_0028' 'MICC_0029'};
% count=1;
% 
% for i_g=1:length(gallery_feature)
%    gallery_feature(i_g).identity_num=i_g; 
% end
% 
% for i_p=1:length(probe_feature)
%     i_p
%     pro_name=probe_feature(i_p).name;
%     if(strcmp(pro_name(1:3),'BRL'))
%         pro_identity=pro_name(1:8);
%         probe_feature(i_p).identity_num=str2num(pro_name(5:8));
%     elseif strcmp(pro_name(1:4),'BU3D')
%         pro_identity=pro_name(1:9);
%         probe_feature(i_p).identity_num=str2num(pro_name(5:8))+124;
%     elseif strcmp(pro_name(1:4),'BU4D')
%         pro_identity=pro_name(1:9);
%         probe_feature(i_p).identity_num=str2num(pro_name(5:8))+124+100;
%     elseif strcmp(pro_name(1:4),'MICC')
%         pro_identity=pro_name(1:9);
%         probe_feature(i_p).identity_num=str2num(pro_name(5:8))+124+100+101;
%     end
%    if ~max(strcmp(test_identity,pro_identity))
%        continue;
%    end
%    pro_index(count)=i_p;
%    count=count+1;
% end
% 
% rank_one_true=0;
% for i_in=1:length(pro_index)
%     i_in
%     for i_gal_fea=1:378
%         feat1=gallery_feature(i_gal_fea).fea';
%         feat2=probe_feature(pro_index(i_in)).fea';
%         scores(i_gal_fea) = distance.compute_cosine_score(feat1, feat2);
%     end
%     [num,index]=max(scores);
%     if index==probe_feature(pro_index(i_in)).identity_num
%         rank_one_true=rank_one_true+1;
%     end
% end
% rank_one_true/length(pro_index)