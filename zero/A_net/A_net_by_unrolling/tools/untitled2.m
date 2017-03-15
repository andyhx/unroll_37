
path = '/home/brl/data/origin_bbox/';
txt_list = dir([path '*.txt']);

for i = 1:length(txt_list)
    i
    fid = fopen([path txt_list(i).name]);
    answ = textscan(fid, '%f%f%f%f');
    bbox(i, :) = [answ{1, 1}, answ{1,2}, answ{1,3}, answ{1,4}];
    name{i} = txt_list(i).name(1:end-4);
end

save('bbox.mat','bbox','name');