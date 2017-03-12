

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

list='train_test_list/0-30.txt';
list1='train_test_list/30-60.txt';
list2='train_test_list/60-90.txt';
% train_list='/home/scw4750/github/r_net/test/train_list.txt';
legend_name={};

%to find best result
weights='/home/scw4750/github/unrolling/zero/Lightencnn/unroll_frontal_data_fine_tuning/snapshot/zhao_softmax_with_low_interval_caffemodel/rnet__iter_8000.caffemodel';
net=caffe.Net(cnnModel,weights,'test');

% %for 0-30
% draw_roc('/home/scw4750/github/r_net/one_dir/',list,net,10);
% legend_name={legend_name{:},'0-30'};
% % 
% for 30-60
% draw_roc('/home/scw4750/github/r_net/one_dir/',list1,net,10);
% legend_name={legend_name{:},'30-60'};

% %for 60-90
draw_roc('/home/scw4750/github/r_net/one_dir/',list2,net,10);
legend_name={legend_name{:},'60-90'};


weights='/home/scw4750/github/unrolling/zero/Lightencnn/final_LightenedCNN_C.caffemodel';
net2=caffe.Net(cnnModel,weights,'test');

%for frontal 0-30
% draw_roc('/home/scw4750/github/ori_frontal/frontal_img/one_dir/',list,net2,10);
% legend_name={legend_name{:},'frontal 0-30'};
% 
% %for frontal 30-60
% draw_roc('/home/scw4750/github/ori_frontal/frontal_img/one_dir/',list1,net2,10);
% legend_name={legend_name{:},'frontal 30-60'};

%for frontal 60-90
draw_roc('/home/scw4750/github/ori_frontal/frontal_img/one_dir/',list2,net2,10);
legend_name={legend_name{:},'frontal 60-90'};

weights='/home/scw4750/github/unrolling/zero/Lightencnn/unroll_frontal_data_fine_tuning/snapshot/frontal/rnet__iter_20000.caffemodel';
net2=caffe.Net(cnnModel,weights,'test');

%for fune tuning frontal 0-30
% draw_roc('/home/scw4750/github/ori_frontal/frontal_img/one_dir/',list,net2,10);
% legend_name={legend_name{:},'tuning-frontal-0-30'};

%for fune tuning frontal 30-60
% draw_roc('/home/scw4750/github/ori_frontal/frontal_img/one_dir/',list1,net2,10);
% legend_name={legend_name{:},'tuning-frontal-30-60'};

%for fune tuning frontal
draw_roc('/home/scw4750/github/ori_frontal/frontal_img/one_dir/',list2,net2,10);
legend_name={legend_name{:},'tuning-frontal-60-90'};

legend(legend_name);
return;





% test_list='/home/scw4750/github/r_net/test/test_list.txt';
% out_dir='/home/scw4750/github/unrolling/zero/Lightencnn/unroll_frontal_data_fine_tuning/compute_roc_by_zhao/train_test_list';
% write_list_with_angle(test_list,out_dir);
function write_list_with_angle(list_loc,out_dir)

fid=fopen(list_loc,'rt');
list=textscan(fid,'%s %d');
fclose(fid);

fid=fopen([out_dir filesep '0-30.txt'],'wt');
fid1=fopen([out_dir filesep '30-60.txt'],'wt');
fid2=fopen([out_dir filesep '60-90.txt'],'wt');

% name=list{1};
% label=list{2};
for i =1:length(list{1})
    i
    name=list{1}{i};
    label=list{1,2}(i);
    id=strfind(name,'_');
    if length(id)==1
        continue;
    end
    index=name(id(end)+1:end-4);
    if strcmp(name(1:3),'BRL')
        range_id=get_range_id_for_brl_micc(str2num(index));
    elseif strcmp(name(1:4),'MICC')
        range_id=get_range_id_for_brl_micc(str2num(index));
    else
        range_id=get_range_id_for_bu3d_bu4d(str2num(index));
    end
    if range_id==0
        fprintf(fid,'%s %d\n',name,label);
    elseif range_id==1
        fprintf(fid1,'%s %d\n',name,label);
    elseif range_id==2
        fprintf(fid2,'%s %d\n',name,label);
    end
end

fclose(fid);fclose(fid1);fclose(fid2);
end

function range_id=get_range_id_for_brl_micc(index)

if index>=25 && index<=48
    range_id=0;
elseif (index>=17&&index<=24) || (index>=48 &&index<=56)
    range_id=1;
else
    range_id=2;
end
end

function range_id=get_range_id_for_bu3d_bu4d(index)

if index>=33 && index<=56
    range_id=0;
elseif (index>=17&&index<=32) || (index>=52 &&index<=72)
    range_id=1;
else
    range_id=2
end
end

