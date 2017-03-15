clear;
landmark_dir='/home/brl/data/landmark';
modified_bbox_dir='/home/brl/data/modified_bbox';
img_info=dir('/home/brl/data/300w/crop_image/*.jpg');
count=1;
error=zeros(6,100);
img_name_landmark(length(img_info)).name=img_info(length(img_info)).name;
for i = 62060:62060%length(img_info)
    i
  img_name_landmark(i).name=img_info(i).name;
  load([landmark_dir '/' img_name_landmark(i).name(1:end-3) 'mat']);
  fid=fopen([modified_bbox_dir '/' img_name_landmark(i).name(1:end-4) '_bbox_modified.txt']);
  modified_bbox=textscan(fid,'%d ');
  fclose(fid);
  topleft_x=single(modified_bbox{1,1}(1,1));
  topleft_y=single(modified_bbox{1,1}(2,1));
  %because we did resize,so the landmark should also resize;
  ratio=100.0/single(modified_bbox{1,1}(3,1));
  img_name_landmark(i).landmark=zeros(3,68);
  if topleft_x < 0 
      topleft_x=0;
      error(1,count)=i;
      count=count+1;
  end
  if  topleft_y<0
      topleft_y=0;
      error(2,count)=i;
      count=count+1;
  end
  for j =1:68
      img_name_landmark(i).landmark(1,j)=(pt2d(1,j)-topleft_x)*ratio;
      img_name_landmark(i).landmark(2,j)=(pt2d(2,j)-topleft_y)*ratio;
      if img_name_landmark(i).landmark(1,j)<0
          img_name_landmark(i).landmark(1,j)=0;
          img_name_landmark(i).landmark(3,j)=1;
          img_name_landmark(i).name(1:end-3)
          error(3,count)=i;
          count=count+1;
      end
      if  img_name_landmark(i).landmark(1,j)>100
          img_name_landmark(i).landmark(1,j)
          img_name_landmark(i).landmark(1,j)=100;
          img_name_landmark(i).landmark(3,j)=1;
          img_name_landmark(i).name(1:end-3)
          error(4,count)=i;
          count=count+1;
      end
      if  img_name_landmark(i).landmark(2,j)<0
          img_name_landmark(i).landmark(2,j)=0;
          img_name_landmark(i).landmark(3,j)=1;   
          img_name_landmark(i).name(1:end-3)
          error(5,count)=i;
          count=count+1;
      end
      if  img_name_landmark(i).landmark(2,j)>100
          img_name_landmark(i).landmark(2,j)=100;
          img_name_landmark(i).landmark(3,j)=1;
          img_name_landmark(i).name(1:end-3)
          error(6,count)=i;
          count=count+1;
      end
  end  
end
crop_img_dir='/home/brl/data/300w/crop_image';
img=imread([crop_img_dir '/' img_name_landmark(i).name]);
pts2d=img_name_landmark(i).landmark;
imshow(img),hold on;
plot(pts2d(1,:), pts2d(2,:), 'g.');
hold off;
%figure, plot_mesh(vertex3d', tri');

%img_para_info=img_para_info(3:end,1);