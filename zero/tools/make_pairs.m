%%ͼ���б��ļ�
imglist_dir = '/home/brl/data/unrolling/imlist.txt';
%%����Ա���1:ratio
ratio = 2;
%% n:ͼ������
n = 4;

fid = fopen(imglist_dir, 'r');
contents = textscan(fid, '%s %d');
paths = contents{1};
labels = contents{2};
same_ptr=1;
diff_ptr=1;
%%positive pairs
for i=1:n
    for j=i+1:n
        if labels(i)==labels(j)
            same_pair{same_ptr} = {paths{i} paths{j} labels(i)};
            same_ptr=same_ptr+1;
        end;
    end;
end;
same_size = length(same_pair);
diff_size = same_size*(ratio+1);
%%�������
p1=randperm(diff_size);
p2=randperm(diff_size);
%%negative pairs
for i=1:diff_size
    k1 = mod(p1(i),n)+1;
    k2 = mod(p2(i),n)+1;
    if(labels(k1)~=labels(k2))
        diff_pair_{diff_ptr} = {paths{k1} labels(k1) paths{k2} labels(k2)};
        diff_ptr=diff_ptr+1;
    end;
end;
if length(diff_pair_) < same_size*ratio
    diff_size = length(diff_pair_);
else
    diff_size = same_size*ratio;
end;
diff_pair = diff_pair_(1:diff_size);

fprintf('same_size:%d    diff_size:%d    ratio:%d\n',same_size,diff_size,ratio);

%%д�ļ�
%%same��ǩΪ1��diff��ǩΪ0
fp1 = fopen('/home/brl/data/unrolling/train.txt','wt');
fp2 = fopen('/home/brl/data/unrolling/train_pair.txt','wt');
p5=randperm(same_size + diff_size);
for i = 1:same_size+diff_size
    k = p5(i);
    if k>same_size
        k = k-same_size;           
        fprintf(fp1,'%s 0\n',diff_pair{k}{1});
        fprintf(fp2,'%s 0\n',diff_pair{k}{3});
    else
        fprintf(fp1,'%s 1\n',same_pair{k}{1});
        fprintf(fp2,'%s 1\n',same_pair{k}{2});
    end;
end;
fclose(fp1);
fclose(fp2);

% %����������
% p3=randperm(same_size);
% p4=randperm(diff_size);
% fsame1 = fopen('same_p1.txt','wt');
% fsame2 = fopen('same_p2.txt','wt');
% fdiff1 = fopen('diff_p1.txt','wt');
% fdiff2 = fopen('diff_p2.txt','wt');
% for i=1:same_size
%     fprintf(fsame1,'%s 1\n',same_pair{p3(i)}{1});
%     fprintf(fsame2,'%s 1\n',same_pair{p3(i)}{2});
% end;
% for i=1:diff_size
%     fprintf(fdiff1,'%s 0\n',diff_pair{p4(i)}{1});
%     fprintf(fdiff2,'%s 0\n',diff_pair{p4(i)}{3});
% end;
% fclose(fsame1);
% fclose(fsame2);
% fclose(fdiff1);
% fclose(fdiff2);



