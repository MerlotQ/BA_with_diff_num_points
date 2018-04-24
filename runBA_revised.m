function hidden_state = runBA_revised(hidden_state, observations, K)
% Update the hidden state, encoded as explained in the problem statement,
% with 20 bundle adjustment iterations.

with_jpattern = true;
% ----- Part3 jacobian pattern begin-----
% spalloc和sparse的作用一样, 类似mask, 非零元素设置为1

% 每个误差项的非零元素（依赖项）对应当前观测它的相机位姿和对应的3D坐标点
if with_jpattern
    num_frames = observations(1);
    num_observations = (numel(observations)-2-num_frames)/3;
    num_error_terms = 2 * num_observations;
    jpattern = spalloc(num_error_terms, numel(hidden_state), ...
        num_error_terms * 9); % 申请的内存数
    
    observation_i = 3;
    error_i = 1; % 这两个相对应是在观测量中的结局    
    for frame_i = 1:num_frames
        num_keypoints_in_frame = observations(observation_i);
        jpattern(error_i:error_i+2*num_keypoints_in_frame-1, ...
            (frame_i-1)*6+1:frame_i*6) = 1;         
        landmark_indices = observations(...
            observation_i+2*num_keypoints_in_frame+1:...
            observation_i+3*num_keypoints_in_frame);    
        for kp_i = 1:length(landmark_indices)  % 对于每一个error term, 其相对应的点坐标 
            jpattern(error_i+(kp_i-1)*2: error_i + kp_i*2-1, ...
                6*num_frames+(landmark_indices(kp_i)-1)*3+1 ... 
              : 6*num_frames+landmark_indices(kp_i)*3) = 1;            
        end
        observation_i = observation_i+3*num_keypoints_in_frame +1;
        error_i = error_i+2*num_keypoints_in_frame;
    end
    figure;
    spy(jpattern);
    
end 
%----- Part3 jacobian pattern end -----

n = observations(1); %帧数
m = observations(2); %路标点index数量 
% 数据格式见 1.4 Data format
obser_idx = []; %向量 存放每一帧数据在obs向量中的位置 
strip_obser_idx = []; % 向量 存放之前帧累加特征点数+1 在二维坐标矩阵中列的索引
obser_cnt = []; % 向量 存放每一帧观测点数 
obser_total = 0; %观测点总数

next_obs_idx = 3; %[numFrames numFeatures [numFeas [points index]] ...]

for ii=1:n
    obser_idx(end+1) = next_obs_idx; 
    strip_obser_idx(end+1) = obser_total+1;
    obser_cnt(end+1) = observations(next_obs_idx);
    obser_total = obser_total + observations(next_obs_idx);
    next_obs_idx = next_obs_idx + 3*observations(next_obs_idx) + 1;
end;

strip_obser_idx(end+1) = obser_total+1; % strip_obser_idx(5) 

obser_image_point = zeros(2, obser_total); % 这个是所有观测的点的坐标
obser_point_idx = zeros(1, obser_total); % 向量 所有观测点的序列号

for j=1:n
    cnt = strip_obser_idx(j+1)-strip_obser_idx(j); %当前特征点数
    % 每一帧变形存储 参考中用的是每一帧求误差项
    obser_image_point(:, strip_obser_idx(j):strip_obser_idx(j+1)-1) = ...
        reshape(observations(obser_idx(j)+1:obser_idx(j)+2*obser_cnt(j)), 2, cnt);
    assert(obser_cnt(j) == cnt)
    % 每一帧的各个点的序列号
    obser_point_idx(:, strip_obser_idx(j):strip_obser_idx(j+1)-1) = ...
        observations(obser_idx(j)+2*obser_cnt(j)+1:obser_idx(j)+3*obser_cnt(j));
end

%根据1.4数据结构obser_image_point是row col的，转换xy需要翻转
obser_image_point = flipud(obser_image_point); % 在相机投影那里翻转了

%x0 = hidden_state(1:n*6); % 只优化位姿扭转向量
 % pos在世界坐标系下特征点3D坐标

options = optimoptions(@lsqnonlin,'Display','iter', 'MaxIterations', 20);
if with_jpattern
   options.JacobPattern = jpattern;
   options.UseParallel = false;
end
hidden_state = lsqnonlin(@error_function,hidden_state,[],[],options); %精简一下

    function E = error_function(xx)
        E = zeros(size(obser_image_point)); 
        for i=1:n
            idx = strip_obser_idx(i):strip_obser_idx(i+1)-1;
            H = twist2HomogMatrix(xx((i-1)*6+1:(i-1)*6+6)); % 从相机到世界坐标系
            H = H^-1;
            % obser_idx = obser_point_idx(:, idx);
            pos = reshape(xx(n*6+1:end), 3, m);
            P = pos(:, obser_point_idx(:, idx)); %避免定义错误
            PC = H(1:3, 1:3)*P + repmat(H(1:3, 4), [1 size(P,2)] );
            PCx = PC(1,:)./PC(3,:);
            PCy = PC(2,:)./PC(3,:);
            %HP = K*(H(1:3, 1:3)*P+H(1:3, 4));
            Proj = K * [PCx;PCy;ones(1, size(PCx,2))];
            %E(1, idx) = HP(2, :) ./ HP(3, :); % y
            %E(2, idx) = HP(1, :) ./ HP(3, :); % x
            E(:,idx) = Proj(1:2,:);
        end;
        E =  E - obser_image_point;
    end
end