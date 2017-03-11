%% image_list dir
imglist_dir = '/home/brl/data/unrolling/imlist.txt';
%% number of images
n = 5;
%% train:val:test = rt:rv:rtt
rt = 1;
rv = 1;
rtt = 1;

%%%%
n_train = floor(n*rt/(rt+rv+rtt));
n_val = floor((n-n_train)*rv/(rv+rtt));
n_test = n-n_train-n_val;
fprintf('sizes of train,val,test sets are : %d   %d   %d\n',n_train,n_val,n_test);
fid = fopen(imglist_dir, 'r');
contents = textscan(fid, '%s %d');
paths = contents{1};
labels = contents{2};
train_ptr=1;
val_ptr=1;
%%����������
p=randperm(n);
ftrain = fopen('train.txt','wt');
fval = fopen('val.txt','wt');
ftest = fopen('test.txt','wt');
for i = 1 : n_train
    fprintf(ftrain,'%s %d\n',paths{p(i)},labels(p(i)));
end;
for i = n_train + 1 : n_train + n_val
    fprintf(fval,'%s %d\n',paths{p(i)},labels(p(i)));
end;
for i = n_train + n_val + 1 : n
    fprintf(ftest,'%s %d\n',paths{p(i)},labels(p(i)));
end;
fclose(ftrain);
fclose(fval);
fclose(ftest);





