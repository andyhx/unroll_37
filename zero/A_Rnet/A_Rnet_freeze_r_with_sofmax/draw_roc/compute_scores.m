
% addpath('/home/scw4750/github/unrolling/zero/Lightencnn/draw_roc');
addpath('/home/scw4750/github/unrolling/zero/tools');
addpath('/home/scw4750/github/unrolling/zero/Lightencnn/tools');
addpath(genpath('/home/scw4750/github/unrolling/matlab'));
addpath('/home/scw4750/github/unrolling/zero/Lightencnn/tools');
cnnModel = '/home/scw4750/github/unrolling/zero/A_Rnet/A_Rnet_freeze_r_with_sofmax/A_Rnet_deploy.prototxt';
use_gpu=true;
if use_gpu
  caffe.set_mode_gpu();
  caffe.set_device(2);
else
    caffe.set_mode_cpu();
end


test_list='/home/scw4750/github/unrolling/zero/A_net/A_net_by_frontal/data/test_list.txt';
legend_name={};


% weights='/home/scw4750/github/unrolling/zero/Lightencnn/frontal_data_tuning/snapshot/rnet__iter_55000.caffemodel';
% net=caffe.Net(cnnModel,weights,'test');
% weight2='/home/scw4750/github/unrolling/zero/A_net/a_net_iter_183000.caffemodel';
% net.copy_from(weight2);
% draw_roc('/home/scw4750/github/unrolling/zero/A_net/A_net_by_frontal/frontal_layer/data/test_true_frontal/',test_list,net,10)
% legend_name={legend_name{:},'300w'};

weights='/home/scw4750/github/unrolling/zero/A_Rnet/A_Rnet_freeze_r_with_sofmax/snapshot/a_rnet_iter_2800.caffemodel';
net=caffe.Net(cnnModel,weights,'test');
draw_roc('/home/scw4750/github/unrolling/zero/A_net/A_net_by_frontal/frontal_layer/data/test_true_frontal/',test_list,net,10)
legend_name={legend_name{:},'300w'};



legend(legend_name);
return ;
% % weights='/home/scw4750/github/unrolling/zero/Lightencnn/unroll_frontal_data_fine_tuning/snapshot/zhao_softmax_with_low_interval_caffemodel/rnet__iter_8000.caffemodel';
% weights='/home/scw4750/github/unrolling/zero/Lightencnn/unroll_frontal_data_fine_tuning/snapshot/centerloss_data_augment/rnet__iter_310000.caffemodel';
% net=caffe.Net(cnnModel,weights,'test');
% 
% %best result for 
% draw_roc('/home/scw4750/github/r_net/one_dir/',test_list,net,10);
% legend_name={legend_name{:},'ours-tuning-augment'};


% % best result for train_list
% draw_roc('/home/scw4750/github/r_net/one_dir/',train_list,net,1000);
% legend_name={legend_name{:},'our-model-best-result-with-train-list'};

% legend(legend_name);
% return;

% %for fune tuning crop frontal
% weights='/home/scw4750/github/unrolling/zero/Lightencnn/snapshot/crop_frontal/rnet__iter_16000.caffemodel';
% net2=caffe.Net(cnnModel,weights,'test');
% draw_roc('/home/scw4750/github/ori_frontal/crop_one_dir/',test_list,net2);
% legend_name={legend_name{:},'tuning-crop-frontal-iter-20000'};


%for fune tuning frontal
weights='/home/scw4750/github/unrolling/zero/Lightencnn/unroll_frontal_data_fine_tuning/snapshot/frontal/rnet__iter_20000.caffemodel';
net=caffe.Net(cnnModel,weights,'test');
draw_roc('/home/scw4750/github/ori_frontal/frontal_img/one_dir/',test_list,net,10);
legend_name={legend_name{:},'frontal-tuning'};

% 
% legend(legend_name);
% return;

weights='/home/scw4750/github/unrolling/zero/Lightencnn/final_LightenedCNN_C.caffemodel';
net=caffe.Net(cnnModel,weights,'test');

% %for our data without fine-tuning model
% draw_roc('/home/scw4750/github/r_net/one_dir/',test_list,net,100);
% legend_name={legend_name{:},'unroll-without-fine-tuning'};

% %for crop_frontal
% 
% draw_roc('/home/scw4750/github/ori_frontal/crop_one_dir/',test_list,net,100);
% legend_name={legend_name{:},'crop-frontal'};

%for origin
draw_roc('/home/scw4750/github/ori_frontal/ori_img/one_dir/',test_list,net,10);
legend_name={legend_name{:},'original'};



%for ours
draw_roc('/home/scw4750/github/r_net/one_dir/',test_list,net,10);
legend_name={legend_name{:},'ours'};



%for frontal
draw_roc('/home/scw4750/github/ori_frontal/frontal_img/one_dir/',test_list,net,10);
legend_name={legend_name{:},'frontal'};



% %to find best result
% weights='/home/scw4750/github/unrolling/zero/Lightencnn/snapshot/zhao_softmax_with_low_interval_caffemodel/rnet__iter_7000.caffemodel';
% net=caffe.Net(cnnModel,weights,'test');
% draw_roc('/home/scw4750/github/r_net/one_dir/',test_list,net,100);
% legend_name={legend_name{:},'fine-tuning-iter-7000'};
% 
% %to find best result
% weights='/home/scw4750/github/unrolling/zero/Lightencnn/snapshot/zhao_softmax_with_low_interval_caffemodel/rnet__iter_10000.caffemodel';
% net=caffe.Net(cnnModel,weights,'test');
% draw_roc('/home/scw4750/github/r_net/one_dir/',test_list,net,100);
% legend_name={legend_name{:},'fine-tuning-iter-10000'};




legend(legend_name);
hold off;
% hold off;
% figure;
% plot_loss();
% hold off;
return;
