
function plot_loss(filename,begin_num)
filename = '/home/scw4750/github/unrolling/zero/Lightencnn/lfw_fine_tuning/log/log.txt';
fid=fopen(filename,'r');
regpat = 'Iteration [0-9]+, loss = [0-9\.]+';
iter = zeros(100000,1);
loss = zeros(100000,1);
p = 1;
while ~feof(fid)
    newline=fgetl(fid);
    o3=regexpi(newline,regpat,'match');
    if ~isempty(o3)
        iterloss = sscanf(o3{1},'Iteration %d, loss = %f');
        iter(p) = iterloss(1);
        loss(p) = iterloss(2);
        p=p+1;
    end;
end;
fclose(fid);
iter = iter(begin_num:p-1);
loss = loss(begin_num:p-1);
plot(iter,loss);
