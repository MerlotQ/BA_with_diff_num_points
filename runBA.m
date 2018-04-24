function hidden_state = runBA(hidden_state, observations, K)
% Update the hidden state, encoded as explained in the problem statement,
% with 20 bundle adjustment iterations.

n = observations(1);
m = observations(2);

obser_idx = [];
strip_obser_idx = [];
obser_cnt = [];
obser_total = 0;

next_obs_idx = 3;
for i=1:n
    obser_idx(end+1) = next_obs_idx;
    strip_obser_idx(end+1) = obser_total+1;
    obser_cnt(end+1) = observations(next_obs_idx);
    obser_total = obser_total + observations(next_obs_idx);
    next_obs_idx = next_obs_idx + 3*observations(next_obs_idx) + 1;
end;
strip_obser_idx(end+1) = obser_total+1;

obser_image_point = zeros(2, obser_total);
obser_point_idx = zeros(1, obser_total);

for i=1:n
    cnt = strip_obser_idx(i+1)-strip_obser_idx(i);
    obser_image_point(:, strip_obser_idx(i):strip_obser_idx(i+1)-1) = ...
        reshape(observations(obser_idx(i)+1:obser_idx(i)+2*obser_cnt(i)), 2, cnt);
    
    obser_point_idx(:, strip_obser_idx(i):strip_obser_idx(i+1)-1) = ...
        observations(obser_idx(i)+2*obser_cnt(i)+1:obser_idx(i)+3*obser_cnt(i));
end

options = optimoptions('lsqnonlin','Display','iter', 'MaxIterations', 20);
hidden_state = lsqnonlin(@error_function,hidden_state,[],[],options);

    function E = error_function(xx)
        E = zeros(size(obser_image_point));        
        for i=1:n
            idx = strip_obser_idx(i):strip_obser_idx(i+1)-1;
            H = twist2HomogMatrix(xx((i-1)*6+1:(i-1)*6+6));
            H = inv(H);
            pos = reshape(xx(n*6+1:end), 3, m);
            obser_idx = obser_point_idx(:, idx);
            P = pos(:, obser_idx);
            HP = K*(H(1:3, 1:3)*P+H(1:3, 4));
            E(1, idx) = HP(2, :) ./ HP(3, :);
            E(2, idx) = HP(1, :) ./ HP(3, :);
        end;
        E = E - obser_image_point;
    end
end