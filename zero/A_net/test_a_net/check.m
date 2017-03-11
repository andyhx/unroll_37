
clear;
close all;
load('Model_Expression.mat');
load('Model_Shape.mat');
load('meanstd.mat');

para=zeros(236,1)';
m=para(229:236)';
p_id=para(1:199)';
p_exp=para(200:228)';

m = m' .* para_std(1:8) + para_mean(1:8);
p_id = p_id' .* para_std(15:213) + para_mean(15:213);
p_exp = p_exp' .* para_std(214:242) + para_mean(214:242);

M(1, :) = m(1:4);
M(2, :) = m(5:8);
%para=[p_id,p_exp,m];
vertex3d = mu_shape + w * p_id' + mu_exp + w_exp * p_exp';
vertextemp=vertex3d;
vertex3d = reshape(vertex3d, 3, length(vertex3d)/3);
vertex4d = vertex3d;
vertex4d(4, :) = 1;
vertex2d = M*vertex4d;


para=zeros(236,1)';
para(1)=para(1)+0.01;
m=para(229:236)';
p_id=para(1:199)';
p_exp=para(200:228)';

m = m' .* para_std(1:8) + para_mean(1:8);
p_id = p_id' .* para_std(15:213) + para_mean(15:213);
p_exp = p_exp' .* para_std(214:242) + para_mean(214:242);

M(1, :) = m(1:4);
M(2, :) = m(5:8);
%para=[p_id,p_exp,m];
pair_vertex3d = mu_shape + w * p_id' + mu_exp + w_exp * p_exp';
pair_vertextemp=pair_vertex3d;
pair_vertex3d = reshape(pair_vertex3d, 3, length(pair_vertex3d)/3);
pair_vertex4d = pair_vertex3d;
pair_vertex4d(4, :) = 1;
pair_vertex2d = M*pair_vertex4d;

norm(vertex2d-pair_vertex2d)

