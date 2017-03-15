clear;
load /home/scw4750/github/a_net/data_(p&m).mat;

addpath(genpath('/home/scw4750/github/unrolling/matlab'));
%addpath(genpath('/home/brl/github/faceRec/toolbox_graph'));
cnnModel = 'deploy.prototxt';
use_basic_model=true;
use_gpu=true;
model='unrolling.caffemodel';
net = caffe.Net(cnnModel,model, 'test');
index=1;
img=data(index).img;
p_shape = data(index).para(1:199);
p_exp = data(index).para(200:228);
p_m = data(index).para(229:236);
net.blobs('img').set_data(img);
net.blobs('p199').set_data(p_shape);
net.blobs('p29').set_data(p_exp);
net.blobs('p8').set_data(p_m);
net.forward_prefilled();
unrolling=net.blobs('unrolling').get_data();
imshow(unrolling);

for i=1:200
   imwrite(data(i).img,['test_img' filesep data(i).name '.jpg']); 
end

