%%write the para in data(p&m).mat to txt
clear
load('../test_a_net/data_2(p&m).mat');
crop_image_dir='/home/brl/data/300w/crop_image';
all_img_info=dir(crop_image_dir);
all_img_name={all_img_info.name}';
all_img_name=all_img_name(3:end);
data_name={data.name};
for i =1:size(all_img_name)
    i
  %name=all_img_name(i);
  id=ismember(data_name,all_img_name{i,1}(1:end-4));
  para=data(id).para;
  fid=fopen([crop_image_dir '/' all_img_name{i,1}(1:end-4) '_para.txt'],'wt');
  %fid=fopen('/home/brl/data/300w/result.txt','wt');
  for j=1:236
     fprintf(fid,'%f ', para(j));
  end
  %fprintf(fid,'%d %d %d %d\n',want_bbox(1:4));
  fclose(fid);
end
