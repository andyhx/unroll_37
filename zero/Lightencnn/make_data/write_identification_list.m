clear;
basic_dir='/home/scw4750/github/ori_frontal';

gallery=dir([basic_dir filesep 'crop_ori_gallery/*.jpg']);
probe=dir([basic_dir filesep 'crop_frontal_img/*.jpg']);


gallery=add_index(gallery);
probe=add_index(probe);

test_identity={'BRL_0003' 'BRL_0006' 'BRL_0007' 'BRL_0008' 'BRL_0013' 'BRL_0014' 'BRL_0015' 'BRL_0016' ...
    'BRL_0002' 'BRL_0004' 'BRL_0005' 'BRL_0023' ...
    'MICC_0001' 'MICC_0002' 'MICC_0003' 'MICC_0004' 'MICC_0028' 'MICC_0029'};
for i_te=1:length(test_identity)
   test_index(i_te)=get_index_by_name(test_identity{i_te}); 
end

train_count=1;
test_count=1;
total_img=[gallery;probe];
r=randperm(length(total_img));
total_img=total_img(r);
train_fid=fopen('train_list.txt','wt');
test_fid=fopen('test_list.txt','wt');
for i_t=1:length(total_img)
    i_t
   if ~max(test_index==total_img(i_t).index)
%        train_list(train_count)=total_img(i_t);
       fprintf(train_fid,'%s %d\n',total_img(i_t).name,total_img(i_t).index-1);
       train_count=train_count+1;
   else
       test_list(test_count)=total_img(i_t);
       fprintf(test_fid,'%s %d\n',total_img(i_t).name,total_img(i_t).index-1);
       test_count=test_count+1;
   end
end
fclose(train_fid);
fclose(test_fid);

%add index to gallery and probe according to their name
function result=add_index(img)

for i=1:length(img)
    img_name=img(i).name;
    img(i).index=get_index_by_name(img_name);
end
result=img;
end

function index=get_index_by_name(name)

img_name=name;
if(strcmp(img_name(1:3),'BRL'))
    index=str2num(img_name(5:8));
elseif strcmp(img_name(1:4),'BU3D')
    index=str2num(img_name(6:9))+124;
elseif strcmp(img_name(1:4),'BU4D')
    index=str2num(img_name(6:9))+124+100;
elseif strcmp(img_name(1:4),'MICC')
    index=str2num(img_name(6:9))+124+100+101;
end
end
