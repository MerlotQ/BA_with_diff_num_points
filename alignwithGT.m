%% 将位姿估计与真值进行匹配，便于比较 处理的情况 
% 1 位姿点集合与真值位姿集合不在一个坐标系下
% 2 两个点集合的点的数量不一致
% 3 采用ICP进行点的匹配

%% 载入估计的点和真值点，并将估计的点减少一些
clear;

% Assumes reference solution of exercise 1 at this location (!).
addpath('../01_camera_projection/code');

hidden_state = load('../data/hidden_state.txt');
observations = load('../data/observations.txt');
num_frames = 150; % 设关键帧只是其子集
K = load('../data/K.txt');
poses = load('../data/poses.txt');
% 'pp' stands for p prime
pp_G_C = poses(:, [4 8 12])';

[hidden_state, observations, pp_G_C] = cropProblem(...
    hidden_state, observations, pp_G_C, num_frames);
[cropped_hidden_state, cropped_observations, ~] = cropProblem(...
    hidden_state, observations, pp_G_C, 4);

%% Compare trajectory to ground truth.
% Remember, V is the "world frame of the visual odometry"...
T_V_C = reshape(hidden_state(1:num_frames*6), 6, []);
p_V_C = zeros(3, num_frames);
for i = 1:num_frames
    single_T_V_C = twist2HomogMatrix(T_V_C(:, i));
    p_V_C(:, i) = single_T_V_C(1:3, end);
end

figure(1);
% ... and G the "world frame of the ground truth".
plot(pp_G_C(3, :), -pp_G_C(1, :));
hold on;
plot(p_V_C(3, :), -p_V_C(1, :));
hold off;
axis equal;
axis([-5 95 -30 10]);
legend('Ground truth', 'Estimate', 'Location', 'SouthWest');

%%
% 首先去掉一些位姿轨迹点，得到关键帧轨迹点
traj=[]; 
for i = 1:size(p_V_C,2)
   if mod(i-1,3)==0 
     traj = [traj p_V_C(:,i)];   
   end
end
gt = pp_G_C;
% 然后将位姿点和真值点分别归一化(求两个点集的中心点)
traj_center = (sum(traj,2))/size(traj,2);
gt_center = (sum(gt,2))/size(gt,2);

figure(2);
plot(gt(3,:),-gt(1,:),'b', gt_center(3),-gt_center(1),'b*');
hold on;
plot(traj(3,:),-traj(1,:),'r', traj_center(3),-traj_center(1),'r*');
hold off;
axis equal;
axis([-5 95 -30 30]);

traj_meand=0; 
for i = 1:size(traj,2)
    traj_meand = traj_meand + norm(traj(:,i)-traj_center); 
end
traj_meand = traj_meand/size(traj,2);
trajs = traj/traj_meand;
traj_center = traj_center/traj_meand;

gt_meand=0;  
for i = 1:size(gt,2)
    gt_meand = gt_meand + norm(gt(:,i)-gt_center); 
end
gt_meand = gt_meand/size(gt,2);
gt = gt/gt_meand; 
gt_center = gt_center/gt_meand;

figure(3);
plot(gt(3,:),-gt(1,:),'b', gt_center(3),-gt_center(1),'b*');
hold on;
plot(trajs(3,:),-trajs(1,:),'r', traj_center(3),-traj_center(1),'r*');
hold off;
% 再进行icp(迭代最近点的求解)
[cor, dist]= closest_two_sets(trajs, gt);
[corr, dista] = closest_two_sets(gt, trajs);
% 按cor来进行匹配 因为实际真值点数目很多，求解align对齐问题需要

%% Align estimate to ground truth.

gt_new = zeros(size(traj));
for i = 1: size(traj,2)
   gt_new(:,i) = pp_G_C(:, cor(i)); 
end

traj_new = alignEstimateToGroundTruth(...
    gt_new, traj);

figure(4);
plot(pp_G_C(3, :), -pp_G_C(1, :));
hold on;
plot(traj(3, :), -traj(1, :));
plot(traj_new(3, :), -traj_new(1, :));
hold off;
axis equal;
axis([-5 95 -30 10]);
legend('Ground truth', 'Original estimate', 'Aligned estimate', ...
    'Location', 'SouthWest');

%% calculate associate trajectory error
ate_error = norm(gt_new - traj_new); 



