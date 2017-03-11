
clear;
load('pos_pair.mat');
load('neg_pair.mat');
r=randperm(size(pos_pair,2));
new_pos_pair=pos_pair(r);
r=randperm(size(neg_pair,2));
new_neg_pair=neg_pair(r);
merge_pair=[pos_pair neg_pair];
r=randperm(size(merge_pair,2));
new_merge_pair=merge_pair(r);
fid=fopen('merge_gallery.txt','wt');
fid_pair=fopen('merge_probe.txt','wt');
for i =1:size(new_merge_pair,2)
   fprintf(fid,'%s %d\n',new_merge_pair(i).img,new_merge_pair(i).label);
   fprintf(fid_pair,'%s %d\n',new_merge_pair(i).img_pair,new_merge_pair(i).label);
end
fclose(fid);
fclose(fid_pair);
save shuffle_merge_pair.mat new_merge_pair;
