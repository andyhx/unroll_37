clear;
jpg_bbox_info=dir('./*.jpg');
jpg_bbox_info=jpg_bbox_info(1:end);
for i =1:size(jpg_bbox_info,1)
    img_name=jpg_bbox_info(i).name;
    img=imread(img_name);
    txt_name=[img_name(1:end-3) 'txt'];
    fid=fopen(txt_name,'rt');
    content=textscan(fid,'%f ');
    bbox=content{1}';
    imshow(img);
    hold on;
    r=rectangle('Position',[bbox(1:2) bbox(3:4)],'Edgecolor','g','LineWidth',3);
    %bbox=[boudingboxes(j,1:2) boudingboxes(j,3:4)-boudingboxes(j,1:2)];
    center_bbox = bbox(1:2) + bbox(3:4)/2;
    temp_dis = max(bbox(3), bbox(4));
    n_bbox(1:2) = center_bbox(1:2)-temp_dis/2;
    n_bbox(3:4) = center_bbox(1:2)+temp_dis/2 - n_bbox(1:2);
    scale=1.3;
    region = enlargingbbox(n_bbox, scale);
    region = round(region);
    
    region(2) = double(max(region(2), 1));
    region(1) = double(max(region(1), 1));
    bottom_y = double(min(region(2) + region(4) - 1, 1080));
    right_x = double(min(region(1) + region(3) - 1, 1920));
    img_region = img(region(2):bottom_y, region(1):right_x, :);
    img_region =  rgb2gray(imresize(img_region, [100, 100]));
    imwrite(img_region,['/home/brl/github/unrolling/zero/test_a_net/test_data' filesep num2str(i) '.jpg']);
end



