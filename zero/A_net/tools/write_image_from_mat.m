% clear;
% load('/home/brl/data/300w/data_3(p&m).mat');

for i =1:length(data)
    i
    img=data(i).img;
    para = data(i).para;
    img_name=data(i).name;
    
    imwrite(img,['/home/brl/data/300w/crop_image' filesep img_name '.jpg']);
    para_name=[img_name '_para.txt'];
    fid=fopen(['/home/brl/data/300w/crop_image' filesep para_name],'wt');
    for j=1:236
     fprintf(fid,'%f ', para(j));
    end
    fclose(fid);
end
