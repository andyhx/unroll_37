feature_file='/home/brl/github/unrolling/zero/feature_lfw.txt';
%the number of the total test images;
num=2;

fid =fopen(feature_file,'r');
features = textscan(fid, '%f');
fea=reshape(features{1,1},256,num);
fea=transpose(fea);
fclose(fid);