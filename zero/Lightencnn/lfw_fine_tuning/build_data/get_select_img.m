

load select_lfw(10images).mat;

path1 = '/home/scw4750/github/LFW_frontal_compare/method2_img/';
path2 = '/home/scw4750/github/LFW_frontal_compare/zhu_img/';
path3 = '/home/scw4750/github/LFW_frontal_compare/our_img/';
des_path1 = './method2/';
des_path2 = './zhu/';
des_path3 = './our/';

for i = 1:length(select_subject_name)
    i
    sub_name = select_subject_name{i};
    for j = 1:select_subject_file_num(i)
        img_name = [sub_name '_' num2str(j, '%04d')];
        copyfile([path1 img_name '.jpg'], [des_path1 img_name '.jpg']);
        copyfile([path2 img_name '_N.jpg'], [des_path2 img_name '.jpg']);
        copyfile([path3 img_name '.jpg.jpg'], [des_path3 img_name '.jpg']);
    end
end

