clear;
load('meanstd.mat');
fid=fopen('/home/brl/para_mean.txt','wt');
for i =1:242
   fprintf(fid,'%f ',para_mean(i)); 
end
fclose(fid);
fid=fopen('/home/brl/para_std.txt','wt');
for i =1:242
   fprintf(fid,'%f ',para_std(i)); 
end
fclose(fid);