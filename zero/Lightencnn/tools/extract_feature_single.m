

function feature=extract_feature_single(img, net)

% addpath(genpath('/home/scw4750/github/unrolling/matlab'));
% 
% net=caffe.Net(cnnModel,weights,'test');
% 
% use_gpu=true;
% 
% if use_gpu
%   caffe.set_mode_gpu();
%   caffe.set_device(2);
% else
%     caffe.set_mode_cpu();
% end

if size(img,3)==3
  img=rgb2gray(img);
end
img=imresize(img,[128,128]);

img=img';

data = zeros(128,128,1,1);
data = single(data);
data(:,:,:,1) = (single(img)/255.0);

net.blobs('image').set_data(data);
net.forward_prefilled();
eltwise_fc1=net.blobs('eltwise_fc1').get_data();
feature=eltwise_fc1';

end

