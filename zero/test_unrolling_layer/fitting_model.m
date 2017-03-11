
load('Model_Expression.mat');
load('Model_Shape.mat');
p = para(1:228);
m = para(229:end);

%% liu
% P = p .* para_std(1:228) + para_mean(1:228);
% m = m .* para_std(229:end) + para_mean(229:end);
% M(1, :) = m(1:4);
% M(2, :) = m(5:8);
% M = M + M0;

%% our test
P = p;  
M(1,:) = m(1:4);
M(2,:) = m(5:8);

vertex3d = mu_shape + w * P(1:199)' + mu_exp + w_exp * P(200:end)';
vertex3d = reshape(vertex3d, 3, length(vertex3d)/3)';



