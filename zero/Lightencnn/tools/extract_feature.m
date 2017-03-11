

function feature=extract_feature(img_dir,img_struct,cnnModel,weights)

addpath(genpath('/home/scw4750/github/unrolling/matlab'));

net=caffe.Net(cnnModel,weights,'test');
use_gpu=true;

if use_gpu
  caffe.set_mode_gpu();
  caffe.set_device(0);
else
    caffe.set_mode_cpu();
end

for i = 1:length(img_struct)
     i
     tic
    img = imread([img_dir filesep img_struct(i).name]);
    if size(img,3)==3
      img=rgb2gray(img);
    end
    img=imresize(img,[128,128]);
    %img=rgb2gray(img);
    img=img';
    data = zeros(128,128,1,1);
    data = single(data);
    data(:,:,:,1) = (single(img)/255.0);
    net.blobs('image').set_data(data);
    net.forward_prefilled();
    eltwise_fc1=net.blobs('eltwise_fc1').get_data();
    img_struct(i).fea=eltwise_fc1';
     toc
end
feature=img_struct;
end

