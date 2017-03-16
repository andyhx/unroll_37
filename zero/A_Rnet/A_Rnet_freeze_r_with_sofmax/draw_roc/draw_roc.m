function draw_roc(rootdir,test_list,net,interval)
% rootdir = '/home/scw4750/github/r_net/one_dir/';

countgallery = 0;
countprobe = 0;
imclass0 = -1;

% test_fid = fopen('/home/scw4750/github/r_net/test/test_list.txt', 'rt');
test_fid=fopen(test_list,'rt');
while 1
    tline = fgetl(test_fid);
    if ~ischar(tline)
        break;
    end
    
    idx = strfind(tline, ' ');
    imname = tline(1:idx-1);
    imclass = str2num(tline(idx+1:end));
    imclass
    if ~exist([rootdir imname])
        continue;
    end
    im = imread([rootdir imname]);
    feature = extract_feature_single_for_arnet(im, net);
    
    if imclass ~= imclass0
        countgallery = countgallery + 1;
        gallery_features{countgallery} = feature;
        gallery_classes(countgallery) = imclass;
        imclass0 = imclass;
    else
        countprobe = countprobe + 1;
        probe_features{countprobe} = feature;
        probe_classes(countprobe) = imclass;
    end
end
fclose(test_fid);

gscores = []; % genuine
iscores = []; % imposter
for gi = 1:countgallery
    gfeature = gallery_features{gi};
    gclass = gallery_classes(gi);
    for pi = 1:countprobe
        pfeature = probe_features{pi};
        pclass = probe_classes(pi);
        
        score = distance.compute_cosine_score(gfeature', pfeature');
        
        if gclass == pclass
            gscores = [gscores; score];
        else
            iscores = [iscores; score];
        end
    end
end

scores = [gscores; iscores];
labels = [ones(length(gscores), 1); zeros(length(iscores), 1)];

roc = evaluation.evaluate('roc', scores, labels);
fprintf('eer:          %f\n', roc.measure);
fprintf('tpr001:       %f\n', roc.extra.tpr001*100);
fprintf('tpr0001:      %f\n', roc.extra.tpr0001*100);
ROC(scores,labels,interval); 
end