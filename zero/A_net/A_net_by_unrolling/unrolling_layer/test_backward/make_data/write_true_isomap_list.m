clear;
true_isomap=dir('../../true_isomap/*.jpg');
test_isomap=dir('../../test_img/*.jpg');
r=randperm(length(true_isomap));
true_isomap=true_isomap(r);
test_isomap=test_isomap(r);
count=0;
fid=fopen('true_isomap_list.txt','wt');
fid1=fopen('test_img_list.txt','wt');
for ite =1:min(length(test_isomap),length(true_isomap))
   if(strcmp(test_isomap(ite).name,true_isomap(ite).name))
       fprintf(fid,'%s 1\n',['true_isomap/' true_isomap(ite).name]);
       fprintf(fid1,'%s 1\n',['test_img/' test_isomap(ite).name]);
       count=count+1;
   end
end
count