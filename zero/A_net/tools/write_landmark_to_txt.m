clear;
imglist=importdata('imglist2.txt');
for i=1:length(imglist)
    i
	%img=imread(imglist{i});
    load([imglist{i}(1:end-3) 'mat']);
    fid=fopen([imglist{i}(1:end-4) '_landmark.txt'],'wt');
    for i=1:68
         fprintf(fid,'%f %f ',pt2d(1,i),pt2d(2,i));
    end
    fclose(fid);
end