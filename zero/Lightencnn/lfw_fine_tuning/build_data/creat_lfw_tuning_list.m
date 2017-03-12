

train_fid=fopen('train_list.txt','wt');
for i = 1:length(select_subject_name)
    sub_name = select_subject_name{i};
    for j = 1:select_subject_file_num(i)
        img_name = [sub_name '_' num2str(j, '%04d')];
        fprintf(train_fid,'%s.jpg %d\n', img_name, i-1);
    end
end
fclose(train_fid);