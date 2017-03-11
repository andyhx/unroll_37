clear;
gallery=dir('/home/scw4750/github/a_rnet/img/gallery/*.jpg');
brl_video=dir('/home/scw4750/github/a_rnet/img/probe/brl_video/*.jpg');
micc_video=dir('/home/scw4750/github/a_rnet/img/probe/micc_video/*.jpg');
%load('train_gallery.mat');
%load('train_brl_probe.mat');
%load('train_micc_probe.mat');
% gallery=train_gallery;
% brl_video=train_brl_probe;
% micc_video=train_micc_probe;
count=0;
tic
for i=1:size(gallery,1)
    img=gallery(i).name;
    for j=1:size(brl_video,1)
        if strcmp(img(1:end-4),brl_video(j).name(1:8))
            fprintf('i:%d  j:%d \n',i,j)
            count=count+1;
            pos_pair(count).img=['gallery/' img];
            pos_pair(count).img_pair=['probe/brl_video/' brl_video(j).name];
            pos_pair(count).label=1;
            pos_pair(count).img_index=i;
            pos_pair(count).img_pair_index=j;
            %fprintf(fid,'%s 1\n',['gallery/' img]);
            %fprintf(fid2,'%s 1\n',['probe/brl_video/' brl_video(j).name]);
        end
    end
    for k=1:size(micc_video,1)
        if strcmp(img(1:end-4),micc_video(k).name(1:9))
            fprintf('i:%d  k:%d \n',i,k)
            count=count+1;
            pos_pair(count).img=['gallery/' img];
            pos_pair(count).img_pair=['probe/micc_video/' micc_video(k).name];
            pos_pair(count).label=1;
            pos_pair(count).img_index=i;
            pos_pair(count).img_pair_index=k+length(brl_video);
            %fprintf(fid,'%s 1\n',['gallery/' img]);
            %fprintf(fid2,'%s 1\n',['probe/brl_video/' micc_video(k).name]);
        end
    end
end
toc
save gallery.mat gallery
probe=[brl_video;micc_video];
save probe.mat probe;
save pos_pair.mat pos_pair;
