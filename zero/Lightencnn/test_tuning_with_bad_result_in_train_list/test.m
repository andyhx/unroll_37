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

train_list='/home/scw4750/github/unrolling/zero/Lightencnn/test_tuning_with_bad_result_in_train_list';




