function p_G_C = alignEstimateToGroundTruth(...
    pp_G_C, p_V_C)
% Returns the points of the estimated trajectory p_V_C transformed into the
% ground truth frame G. The similarity transform Sim_G_V is to be chosen
% such that it results in the lowest error between the aligned trajectory
% points p_G_C and the points of the ground truth trajectory pp_G_C. All
% matrices are 3xN.

% x0 = [1; zeros(6, 1)]; %顺序应该是[扭转向量;1] 这样写也行
x0 = [1;HomogMatrix2twist(eye(4))]; %[ scale; twist ]
% 添加option 
options = optimoptions(@lsqnonlin,'Display','iter');
x = lsqnonlin(@error_function,x0,[],[],options);

H = twist2HomogMatrix(x(2:7));
p_G_C = x(1)*H(1:3, 1:3)*p_V_C+H(1:3, 4);

    function E = error_function(xx)   %输入就是要拟合的参数
        HH = twist2HomogMatrix(xx(2:7));
        % E = xx(1)*HH(1:3, 1:3)*p_V_C + HH(1:3,4) - pp_G_C;  % HH(1:3,4) 维数不对
        num_frames = size(p_V_C, 2);
        E = xx(1)*HH(1:3, 1:3)*p_V_C + repmat(HH(1:3,4), [1 num_frames])- pp_G_C;
    end
end