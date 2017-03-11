clear;
gallery=dir('../img/gallery/*.jpg');
brl_video=dir('../img/probe/brl_video/*.jpg');
micc_video=dir('../img/probe/micc_video/*.jpg');
count=1;
disposed_id={'BRL_0003' 'BRL_0006' 'BRL_0007' 'BRL_0008'...
    'BRL_0016' 'BRL_0013' 'BRL_0014' 'BRL_0015'...
    'BRL_0002' 'BRL_0004' 'BRL_0005' 'BRL_0023'...
    'MICC_0001' 'MICC_0002' 'MICC_0003' 'MICC_0004'...
    'MICC_0028' 'MICC_0029'};

for i=1:length(gallery)
    if ~isempty(find(strcmp(disposed_id,gallery(i).name(1:end-4)), 1))
        disposed_index(count)=i;
        count=count+1;
    end
end
train_gallery=gallery;
train_gallery(disposed_index)=[];


count=1;
disposed_index=[];
for i=1:length(brl_video)
    if  ~isempty(find(strcmp(disposed_id,brl_video(i).name(1:8)), 1))
            dispose_index(count)=i;
           count=count+1;
    end
end
train_brl_probe=brl_video;
train_brl_probe(dispose_index)=[];

count=1;
disposed_index=[];
for i=1:length(micc_video)
    if  ~isempty(find(strcmp(disposed_id,micc_video(i).name(1:9)), 1))
            dispose_index(count)=i;
           count=count+1;
    end
end
train_micc_probe=micc_video;
train_micc_probe(dispose_index)=[];

save train_gallery.mat train_gallery;
save train_brl_probe.mat train_brl_probe;
save train_micc_probe.mat train_micc_probe;