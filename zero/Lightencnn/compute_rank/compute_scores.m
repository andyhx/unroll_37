
addpath('/home/scw4750/github/unrolling/zero/Lightencnn/draw_roc');
addpath('/home/scw4750/github/unrolling/zero/tools');
addpath('/home/scw4750/github/unrolling/zero/Lightencnn/tools');
addpath(genpath('/home/scw4750/github/unrolling/matlab'));
cnnModel = '/home/scw4750/github/unrolling/zero/R_net/deploy.prototxt';
use_gpu=true;
if use_gpu
  caffe.set_mode_gpu();
  caffe.set_device(2);
else
    caffe.set_mode_cpu();
end

test_list='/home/scw4750/github/r_net/test/test_list.txt';
legend_name={};

%to find best result
weights='/home/scw4750/github/unrolling/zero/Lightencnn/final_LightenedCNN_C.caffemodel';
net=caffe.Net(cnnModel,weights,'test');
% draw_roc('/home/scw4750/github/r_net/one_dir/',test_list,net);
draw_roc('/home/scw4750/github/r_net/crop_one_dir/',test_list,net);
legend_name={legend_name{:},'crop'};

%to find best result
weights='/home/scw4750/github/unrolling/zero/Lightencnn/final_LightenedCNN_C.caffemodel';
net=caffe.Net(cnnModel,weights,'test');
draw_roc('/home/scw4750/github/r_net/one_dir/',test_list,net);
% draw_roc('/home/scw4750/github/r_net/crop_one_dir/',test_list,net);
legend_name={legend_name{:},'without crop'};

legend(legend_name);
return;

%to find best result
weights='/home/scw4750/github/unrolling/zero/Lightencnn/snapshot/zhao_softmax_with_low_interval_caffemodel/rnet__iter_8000.caffemodel';
net=caffe.Net(cnnModel,weights,'test');
draw_roc('/home/scw4750/github/r_net/one_dir/',test_list,net);
% draw_roc('/home/scw4750/github/r_net/crop_one_dir/',test_list,net);
legend_name={legend_name{:},'our-model-best-result'};


%for our data using fine-tuning model using centerloss
weights='/home/scw4750/github/unrolling/zero/Lightencnn/final_LightenedCNN_C.caffemodel';net=caffe.Net(cnnModel,weights,'test');
draw_roc('/home/scw4750/github/r_net/one_dir/',test_list,net);
legend_name={legend_name{:},'center-iter-39600'};

legend(legend_name);
return;

% %for fune tuning crop frontal
% weights='/home/scw4750/github/unrolling/zero/Lightencnn/snapshot/crop_frontal/rnet__iter_16000.caffemodel';
% net2=caffe.Net(cnnModel,weights,'test');
% draw_roc('/home/scw4750/github/ori_frontal/crop_one_dir/',test_list,net2);
% legend_name={legend_name{:},'tuning-crop-frontal-iter-20000'};


%for fune tuning frontal
weights='/home/scw4750/github/unrolling/zero/Lightencnn/snapshot/frontal/rnet__iter_20000.caffemodel';
net2=caffe.Net(cnnModel,weights,'test');
draw_roc('/home/scw4750/github/ori_frontal/frontal_img/one_dir/',test_list,net2);
legend_name={legend_name{:},'tuning-frontal-iter-20000'};
% 
% legend(legend_name);
% return;

weights='/home/scw4750/github/unrolling/zero/Lightencnn/final_LightenedCNN_C.caffemodel';
net2=caffe.Net(cnnModel,weights,'test');

% %for our data without fine-tuning model
% draw_roc('/home/scw4750/github/r_net/one_dir/',test_list,net2);
% legend_name={legend_name{:},'unroll-without-fine-tuning'};

%for crop_frontal

draw_roc('/home/scw4750/github/ori_frontal/crop_one_dir/',test_list,net2);
legend_name={legend_name{:},'crop-frontal'};

%for frontal
draw_roc('/home/scw4750/github/ori_frontal/frontal_img/one_dir/',test_list,net2);
legend_name={legend_name{:},'frontal'};

%for origin
draw_roc('/home/scw4750/github/ori_frontal/ori_img/one_dir/',test_list,net2);
legend_name={legend_name{:},'original'};

% %to find best result
% weights='/home/scw4750/github/unrolling/zero/Lightencnn/snapshot/zhao_softmax_with_low_interval_caffemodel/rnet__iter_7000.caffemodel';
% net=caffe.Net(cnnModel,weights,'test');
% draw_roc('/home/scw4750/github/r_net/one_dir/',test_list,net);
% legend_name={legend_name{:},'fine-tuning-iter-7000'};
% 
% %to find best result
% weights='/home/scw4750/github/unrolling/zero/Lightencnn/snapshot/zhao_softmax_with_low_interval_caffemodel/rnet__iter_10000.caffemodel';
% net=caffe.Net(cnnModel,weights,'test');
% draw_roc('/home/scw4750/github/r_net/one_dir/',test_list,net);
% legend_name={legend_name{:},'fine-tuning-iter-10000'};




legend(legend_name);
hold off;
% hold off;
% figure;
% plot_loss();
% hold off;
return;
% function draw_roc(rootdir,test_list,net)
% % rootdir = '/home/scw4750/github/r_net/one_dir/';
% 
% countgallery = 0;
% countprobe = 0;
% imclass0 = -1;
% 
% % test_fid = fopen('/home/scw4750/github/r_net/test/test_list.txt', 'rt');
% test_fid=fopen(test_list,'rt');
% while 1
%     tline = fgetl(test_fid);
%     if ~ischar(tline)
%         break;
%     end
%     
%     idx = strfind(tline, ' ');
%     imname = tline(1:idx-1);
%     imclass = str2num(tline(idx+1:end));
%     imclass
%     im = imread([rootdir imname]);
%     feature = extract_feature_single(im, net);
%     
%     if imclass ~= imclass0
%         countgallery = countgallery + 1;
%         gallery_features{countgallery} = feature;
%         gallery_classes(countgallery) = imclass;
%         imclass0 = imclass;
%     else
%         countprobe = countprobe + 1;
%         probe_features{countprobe} = feature;
%         probe_classes(countprobe) = imclass;
%     end
% end
% fclose(test_fid);
% 
% gscores = []; % genuine
% iscores = []; % imposter
% for gi = 1:countgallery
%     gfeature = gallery_features{gi};
%     gclass = gallery_classes(gi);
%     for pi = 1:countprobe
%         pfeature = probe_features{pi};
%         pclass = probe_classes(pi);
%         
%         score = distance.compute_cosine_score(gfeature', pfeature');
%         
%         if gclass == pclass
%             gscores = [gscores; score];
%         else
%             iscores = [iscores; score];
%         end
%     end
% end
% 
% scores = [gscores; iscores];
% labels = [ones(length(gscores), 1); zeros(length(iscores), 1)];
% 
% roc = evaluation.evaluate('roc', scores, labels);
% fprintf('eer:          %f\n', roc.measure);
% fprintf('tpr001:       %f\n', roc.extra.tpr001*100);
% fprintf('tpr0001:      %f\n', roc.extra.tpr0001*100);
% ROC(scores,labels,10); 
% end