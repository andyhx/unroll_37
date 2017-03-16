
ori_img_path = '/home/scw4750/github/r_net/RNpre_probe_img/';
img_list = dir([ori_img_path '*.jpg']);

for i = 1:length(img_list)
    i
    img_name = img_list(i).name(1:end-4);
    img = imread([ori_img_path img_list(i).name]);
    imwrite(img, ['./anugment_img/' img_name '_01.jpg']);
    for j = 1:9
        tran_rand = randperm(20, 4);
        crop_img = img(tran_rand(1):size(img,2)-tran_rand(2), ...
            tran_rand(3):size(img,1)-tran_rand(4), :);
        
        crop_img = imresize(crop_img, [128, 128]);
        imwrite(crop_img, ['./anugment_img/' img_name '_' num2str(j+1, '%02d') '.jpg']);
    end
end
