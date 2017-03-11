
% clear;
%%
clear;
load('para_esti.mat');
load('Model_Expression.mat');
load('Model_Shape');
load('para_242.mat');
% load('..\meanstd.mat');
mean_shape = double(mu_exp+mu_shape);
n_times = norm(mean_shape)/100;
mean_shape = (mean_shape)/n_times;

%% norm the expression basis,
for i =  1:29
    norm_exp(i) = norm(w_exp(:,i));
    w_exp(:,i) = w_exp(:,i)./norm_exp(i);
end


m = para_esti(229:end)';
p_id = para_esti(1:199)';
p_exp = para_esti(200:228)';

m = m' .* para_std(1:8) + para_mean(1:8);
p_id = p_id' .* para_std(15:213) + para_mean(15:213);
p_exp = p_exp' .* para_std(214:242) + para_mean(214:242);
p_id = p_id./n_times;
p_exp = p_exp.*norm_exp;
p_exp = p_exp./n_times;
m(1:3) = m(1:3).*n_times;
m(5:7) = m(5:7).*n_times;

M(1, 1:4) = m(1:4);
M(2, 1:4) = m(5:8);

vertex3d = mean_shape + w * p_id' + w_exp * p_exp';
vertex3d = reshape(vertex3d, 3, length(vertex3d)/3);
vertex4d = vertex3d;
vertex4d(4, :) = 1;
vertex2d = M*vertex4d;

duv_grad = zeros(53215, 236*2);
for k = 1:53215
    k
    a_u_id = m(1)*w((k-1)*3+1,:) + m(2)*w((k-1)*3+2,:) + m(3)*w((k-1)*3+3,:);
    a_v_id = m(5)*w((k-1)*3+1,:) + m(6)*w((k-1)*3+2,:) + m(7)*w((k-1)*3+3,:);
    a_u_exp = m(1)*w_exp((k-1)*3+1,:) + m(2)*w_exp((k-1)*3+2,:) + m(3)*w_exp((k-1)*3+3,:);
    a_v_exp = m(5)*w_exp((k-1)*3+1,:) + m(6)*w_exp((k-1)*3+2,:) + m(7)*w_exp((k-1)*3+3,:);
    
    a_u_m(1) = mean_shape((k-1)*3+1) + p_id *w((k-1)*3+1,:)' + p_exp *w_exp((k-1)*3+1,:)';
    a_u_m(2) = mean_shape((k-1)*3+2) + p_id *w((k-1)*3+2,:)' + p_exp *w_exp((k-1)*3+1,:)';
    a_u_m(3) = mean_shape((k-1)*3+3) + p_id *w((k-1)*3+3,:)' + p_exp *w_exp((k-1)*3+1,:)';
    a_u_m(4) = 1;
    a_u_m(5) = 0;
    a_u_m(6) = 0;
    a_u_m(7) = 0;
    a_u_m(8) = 0;
    
    a_v_m(1) = 0;
    a_v_m(2) = 0;
    a_v_m(3) = 0;
    a_v_m(4) = 0;
    a_v_m(5) = mean_shape((k-1)*3+1) + p_id *w((k-1)*3+1,:)' + p_exp *w_exp((k-1)*3+1,:)';
    a_v_m(6) = mean_shape((k-1)*3+2) + p_id *w((k-1)*3+2,:)' + p_exp *w_exp((k-1)*3+1,:)';
    a_v_m(7) = mean_shape((k-1)*3+3) + p_id *w((k-1)*3+3,:)' + p_exp *w_exp((k-1)*3+1,:)';
    a_v_m(8) = 1;
    
    duv_grad(k, 1:236) = [a_u_id a_u_exp a_u_m];
    duv_grad(k, 237:end) = [a_v_id a_v_exp a_v_m];
end


