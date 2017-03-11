clear;
basic_dir='//home/scw4750/github/r_net';
gallery=dir([basic_dir '/gallery/*.jpg']);
probe=dir('/home/scw4750/github/r_net/RNpre_probe_img/*.jpg');

% load('train_gallery.mat');
% load('train_brl_probe.mat');
% load('train_micc_probe.mat');
% gallery=train_gallery;
% brl_video=train_brl_probe;
% micc_video=train_micc_probe;
count=0;
tic
test_identity={'BRL_0003' 'BRL_0006' 'BRL_0007' 'BRL_0008' 'BRL_0013' 'BRL_0014' 'BRL_0015' 'BRL_0016' ...
    'BRL_0002' 'BRL_0004' 'BRL_0005' 'BRL_0023' ...
    'MICC_0001' 'MICC_0002' 'MICC_0003' 'MICC_0004' 'MICC_0028' 'MICC_0029'};
make_val_data=true;
if make_val_data ~=true
    for i_g=1:size(gallery,1)
        i_g
        gal_name=gallery(i_g).name;
        if(strcmp(gal_name(1:3),'BRL'))
            gal_identity=gal_name(1:8);
        else
            gal_identity=gal_name(1:9);
        end
        if max(strcmp(test_identity,gal_identity))
            continue;
        end
        for i_p=1:size(probe,1)
            i_p
            pro_name=probe(i_p).name;
            if(strcmp(pro_name(1:3),'BRL'))
                pro_identity=pro_name(1:8);
            else
                pro_identity=pro_name(1:9);
            end
            if max(strcmp(test_identity,pro_identity))
                continue;
            end
            if ~strcmp(gal_identity,pro_identity)
                %fprintf('%d \n',j)
                if rand()<0.003
                    count=count+1;
                    neg_pair(count).img=['gallery/' gal_name];
                    neg_pair(count).img_pair=['RNpre_probe_img/' pro_name];
                    neg_pair(count).label=0;
                    neg_pair(count).img_index=i_g;
                    neg_pair(count).img_pair_index=i_p;
                    %fprintf(fid,'%s 1\n',['gallery/' img]);
                    %fprintf(fid2,'%s 1\n',['probe/brl_video/' brl_video(j).name]);
                end
            end
        end
    end
    toc
    save neg_pair.mat neg_pair;
else
    for i_g=1:size(gallery,1)
        i_g
        gal_name=gallery(i_g).name;
        if(strcmp(gal_name(1:3),'BRL'))
            gal_identity=gal_name(1:8);
        else
            gal_identity=gal_name(1:9);
        end
        if ~max(strcmp(test_identity,gal_identity))
            continue;
        end
        for i_p=1:size(probe,1)
            i_p
            pro_name=probe(i_p).name;
            if(strcmp(pro_name(1:3),'BRL'))
                pro_identity=pro_name(1:8);
            else
                pro_identity=pro_name(1:9);
            end
            if ~max(strcmp(test_identity,pro_identity))
                continue;
            end
            if ~strcmp(gal_identity,pro_identity)
                %fprintf('%d \n',j)
                if rand()<0.003
                    count=count+1;
                    val_neg_pair(count).img=['gallery/' gal_name];
                    val_neg_pair(count).img_pair=['RNpre_probe_img/' pro_name];
                    val_neg_pair(count).label=0;
                    val_neg_pair(count).img_index=i_g;
                    val_neg_pair(count).img_pair_index=i_p;
                    %fprintf(fid,'%s 1\n',['gallery/' img]);
                    %fprintf(fid2,'%s 1\n',['probe/brl_video/' brl_video(j).name]);
                end
            end
        end
    end
    toc
    save val_neg_pair.mat val_neg_pair;
end