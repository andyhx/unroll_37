
dbstop if error
clear
%%
load('/home/brl/github/unrolling/zero/test_weight_pm_layer/Model_Expression.mat');
load('/home/brl/github/unrolling/zero/test_weight_pm_layer/Model_Shape.mat');
load('para_wpdc.mat');

for i = 1:2
    %%  ground truth
    p_id_gt = para_gt(i,1:199);
    p_exp_gt = para_gt(i,200:228);
    m_gt = para_gt(i,229:end);
    
    M_gt(1, 1:4) = m_gt(1:4);
    M_gt(2, 1:4) = m_gt(5:8);
    
    vertex3d_gt = mu_shape + w * p_id_gt' + mu_exp + w_exp * p_exp_gt';
    vertex3d_gt = reshape(vertex3d_gt, 3, length(vertex3d_gt)/3);
    vertex4d_gt = vertex3d_gt;
    vertex4d_gt(4, :) = 1;
    vertex2d_gt = M_gt*vertex4d_gt;
    
    %     figure, plot(vertex2d_gt(1,:), vertex2d_gt(2,:), 'g.');
    
    %% estimated
    for k = 1:236
        
        para_new = para_gt(i,:);
        para_new(k) = para_es(i,k);
        
        p_id_es = para_new(1:199);
        p_exp_es = para_new(200:228);
        m_es = para_new(229:end);
        
        M_es(1, 1:4) = m_es(1:4);
        M_es(2, 1:4) = m_es(5:8);
        
        vertex3d_es = mu_shape + w * p_id_es' + mu_exp + w_exp * p_exp_es';
        vertex3d_es = reshape(vertex3d_es, 3, length(vertex3d_es)/3);
        vertex4d_es = vertex3d_es;
        vertex4d_es(4, :) = 1;
        vertex2d_es = M_es*vertex4d_es;
        
        %figure, plot(vertex2d_es(1,:), vertex2d_es(2,:), 'g.');
        
        U_dis(i,k) = norm(vertex2d_gt-vertex2d_es);
    end
end

