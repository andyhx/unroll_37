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
c_01=0;
c_c1=0;
c_c2=0;
c_c3=0;
c_c4=0;
c_ic=0;
c_i=0;
c_o=0;
tic
for i=1:size(gallery,1)
    do_write=false;
    img=gallery(i).name;
    if strcmp(img(1:3),'BRL')
      identity=img(1:8);
    else
      identity=img(1:9);
    end
    for j=1:size(brl_video,1)
        fprintf('i:%d  j:%d \n',i,j)
        do_write=false;
        if ~strcmp(identity,brl_video(j).name(1:8))
            scene=brl_video(j).name(10:11);
            if strcmp(scene,'01')
                if rand()<0.05
                  do_write=true;
                  c_01=c_01+1;
                end
            elseif strcmp(scene,'C1')
                if rand()<0.01
                    do_write=true;
                    c_c1=c_c1+1;
                end
            elseif strcmp(scene,'C2')
                if rand()<0.005
                    do_write=true;
                    c_c2=c_c2+1;
                end
            elseif strcmp(scene,'C3')
                if rand()<0.01
                    do_write=true;
                    c_c3=c_c3+1;
                end
            elseif strcmp(scene,'C4')
                if rand()<0.005
                    do_write=true;
                    c_c4=c_c4+1;
                end
            end
            if do_write==true
              count=count+1;
              neg_pair(count).img=['gallery/' img];
              neg_pair(count).img_pair=['probe/brl_video/' brl_video(j).name];
              neg_pair(count).label=0;
              neg_pair(count).img_index=i;
              neg_pair(count).img_pair_index=j;
              %fprintf(fid,'%s 0\n',['gallery/' img]);
              %fprintf(fid2,'%s 0\n',['probe/brl_video/' brl_video(j).name]);
            end
        end
    end
    for k=1:size(micc_video,1)
        fprintf('i:%d  k:%d \n',i,k)
        if ~strcmp(identity,micc_video(k).name(1:9))
            scene=micc_video(k).name(11:12);
            do_write=false;
            if strcmp(scene,'IC')
                if rand()<0.00025
                    do_write=true;
                    c_ic=c_ic+1;
                end
            elseif strcmp(scene,'I_')
                if rand()<0.00005
                    do_write=true;
                    c_i=c_i+1;
                end
            elseif strcmp(scene,'O_')
                if rand()<0.00015
                    do_write=true;
                    c_o=c_o+1;
                end
            end
            if do_write==true
             count=count+1;
             neg_pair(count).img=['gallery/' img];
             neg_pair(count).img_pair=['probe/micc_video/' micc_video(k).name];
             neg_pair(count).label=0;
             neg_pair(count).img_index=i;
             neg_pair(count).img_pair_index=k+length(brl_video);
%              fprintf(fid,'%s 0\n',['gallery/' img]);
%              fprintf(fid2,'%s 0\n',['probe/brl_video/' micc_video(k).name]);
            end
        end
    end
end
toc
save neg_pair.mat neg_pair;
