%Initializing multi-robot system with safety barrier
%certificates (SBC) directly from Robotarium to avoid inter-robot collisions without consieration of
%uncertainty
%Wenhao Luo (whluo12@gmail.com)
%Last modified: 5/25/2020

%code adapted from Georgia Tech Robotarium Repo at 
%https://github.com/robotarium/robotarium-matlab-simulator



figure(99)
clf
% close all;
% clear all;
% Get Robotarium object used to communicate with the robots/simulator
rb = RobotariumBuilder();

% Get the number of available agents from the Robotarium.  We don't need a
% specific value for this algorithm
N = rb.get_available_agents(); 

% Set the number of agents and whether we would like to save data.  Then,
% build the Robotarium simulator object!
N = 23;
confidence_level = 0.9; % not used by SBC
SafetyRadius = 0.2; % should manually keep consistent with the initial value in ARobotarium
r = rb.set_number_of_agents(N).set_save_data(false).build();

r.set_dynamics_flag(true);
r.set_ghost_flag(true);
% rng(99);
x_rand_span_x = 0.02*randi([3 4],1,N); % setting up position error range for each robot, rand_span serves as the upper bound of uncertainty for each of the robot
x_rand_span_y = 0.02*randi([1 4],1,N); %
v_rand_span = 0.005*ones(2,N); % setting up velocity error range for each robot

r.set_ghost_error_box([x_rand_span_x;x_rand_span_y]);
r.set_vel_error(zeros(2,N));

% Initialize x so that we don't run into problems later.  This isn't always
% necessary
x = r.get_poses();
ctrl_flag = [1 10 2 20 3 30 4 40 5 50 6 60 7 70 8 80 9 90 10 100 11 110 12 120 13 130 14 140 15 150 16 160 17 170 18 180 19 190 20 200 21 210 22 220 23 230 24 240 25 250];

r.set_radius(SafetyRadius);
r.set_ghost_poses_error(zeros(2,N));
r.step();
r.set_ctrl_flag(ctrl_flag);


%Hallway
%x = [-2.5 -2.5 2.5 2.4 -1.3 -1.7 -2.1 -2.5 -2.9 -0.9 -0.5 0.0 0.4 0.8 1.2 1.6 2.0 2.4 -1.3 -1.7 -2.1 -2.5 -2.9 -0.9 -0.5 0.0 0.4 0.8 1.2 1.6 2.0;...
 %  -0.4 0.4 0.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0;...
  %  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];

%goal_condition = x(1:2,[2 1 4 3 6 5 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50]); % robots move to swap their positions
p = [-1.0 -1.0;...
    2.0 2.0];
 %p = [0.0;...
  %   2.5];
  q = [4.0 4.0 4.0 4.0;...
       4.0 4.0 4.0 4.0];


%hallway goal condition
goal_condition = [p(1:2,[1 2 3]) q(1:2,[4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4])];

%hallway gp
gp = [p(1:2,[1 2 3]) q(1:2,[4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4])];

r.set_poses(x);

hold on;

% Plot the goal circle
%goal_center = [-1.3 -1.1 -0.9 -0.7 -0.5 -1.5 -1.7 -1.9 -2.1 -2.3 -2.5 -2.7 -2.9 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 -0.5 -0.5 -0.5 -0.5 1.0 1.0 1.0 1.0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.2 0.4 0.6 0.8 0.2 0.4 0.6 0.8];
%goal_radius = 0.1;
%for i = 1:32
 %   viscircles(goal_center(:, i)', goal_radius, 'Color', 'k', 'LineWidth', 3);
%end
%goal_center = [-1.3 -1.1 -0.9 -0.7 -0.5 -1.5 -1.7 -1.9 -2.1 -2.3 -2.5 -2.7 -2.9 -0.3 -0.1 0.1 0.3 0.5 0.7 0.9 2.0 2.2 2.4 2.6 2.8 3.0 0.9 0.9 0.9 0.9 2.0 2.0 2.0 2.0 ; -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0  -1.2 -1.4 -1.6 -1.8 -1.2 -1.4 -1.6 -1.8];
%goal_radius = 0.1;
%for i = 1:34
 %   viscircles(goal_center(:, i)', goal_radius, 'Color', 'k', 'LineWidth', 3);
%end

%goal_center = [0.0; 0.0];
%goal_radius = 1.0;
%viscircles(goal_center', goal_radius, 'Color', 'k', 'LineWidth', 3 );

%goal_center = [-1.3; -0.5];
%goal_radius = 0.3;
%viscircles(goal_center', goal_radius, 'Color', 'g', 'LineWidth', 3 );

%goal_center = [1.8; -0.5];
%goal_radius = 0.3;
%viscircles(goal_center', goal_radius, 'Color', 'b', 'LineWidth', 3 );

handle_targets_dummy = cell(0,1);
for ijk_goal = 1:N
    
    if ctrl_flag(ijk_goal) == 0 % light black
        ctrl_color = 'flat';
        radius_color = [0 0 0];
    elseif ctrl_flag(ijk_goal) == 3   % MSFT Blue
        ctrl_color = [0 161 241]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 2   % MSFT Green
        ctrl_color = [124 187 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 4   % MSFT Yellow
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;        
    elseif ctrl_flag(ijk_goal) == 5   
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
        
    elseif ctrl_flag(ijk_goal) == 6
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 7
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 8
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 9
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 10
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 11
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 12
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 13
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 14
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 15
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 16
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 17
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 18
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 19
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 20
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 21
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 22
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 23
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 24
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 25
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 1   % MSFT Red
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 10   % Dark Red
        ctrl_color = [255 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 20   % Dark Green
        ctrl_color = [0 102 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 30   % Dark Blue
        ctrl_color = [0 0 255]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 40   
        ctrl_color = [60 45 25]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 50
        ctrl_color = [160 145 125]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 60
        ctrl_color = [160 145 125]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 70
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 80
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 90
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 100
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 110
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 120
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 130
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 140
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 150
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 160
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 170
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 180
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 190
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 200
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 210
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 220
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 230
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    elseif ctrl_flag(ijk_goal) == 240
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;      
    elseif ctrl_flag(ijk_goal) == 250
        ctrl_color = [0 0 0]/255;
        radius_color = ctrl_color;
    end
    
    x_pri = [goal_condition(1, ijk_goal)-0.04 goal_condition(1, ijk_goal)+0.04 goal_condition(1, ijk_goal)+0.04 goal_condition(1, ijk_goal)-0.04];
    y_pri = [goal_condition(2,ijk_goal)-0.04 goal_condition(2,ijk_goal)-0.04 goal_condition(2,ijk_goal)+0.04 goal_condition(2,ijk_goal)+0.04];
    
    text(goal_condition(1, ijk_goal)-0.1, goal_condition(2,ijk_goal)+0.3, num2str(ijk_goal),'FontSize',30, 'Color','r');    

    handle_targets_dummy{ijk_goal} = patch(x_pri,y_pri,ctrl_color,'EdgeColor','none','FaceAlpha',0.6);

end

% Create a barrier certificate so that the robots don't collide
si_barrier_certificate = create_si_barrier_certificate('SafetyRadius', 2*SafetyRadius);% 
si_to_uni_dynamics = create_si_to_uni_mapping2();
fun_rand = @(a,B) [a.*B(1,:);a.*B(2,:)];

        
%Get randomized initial conditions in the robotarium arena
initial_conditions = generate_initial_conditions(N, 'Width', r.boundaries(2)-r.boundaries(1)-0.1, 'Height', r.boundaries(4)-r.boundaries(3)-0.1, 'Spacing', 0.2);

% We'll make the rotation error huge so that the initialization checker
% doesn't care about it
args = {'PositionError', 0.01, 'RotationError', 50};
init_checker = create_is_initialized(args{:});
controller = create_si_position_controller();
timer_to_stop = 2000;
timer_count = 1;
agent_positions = zeros(timer_to_stop, N, 2);

record_video_flag = true;
if record_video_flag
    writerObj = VideoWriter('Hallway_GT_01.mp4', 'MPEG-4');
    open(writerObj);
    set(gca,'nextplot','replacechildren');
    set(gcf,'Renderer','zbuffer');
    r.set_video_obj(writerObj);
end

rng(150)

font_size = 25;



%% prepare for computing the performance-related metrics
min_bot_dist_hist = zeros(timer_to_stop,1);
min_bot_dist_observe_hist = zeros(timer_to_stop,1);
min_bot_safe_ratio_hist = zeros(timer_to_stop,1);

pertub_bot_hist = zeros(timer_to_stop,1);
perf_bot_hist = zeros(timer_to_stop,1);
dxu_data = zeros(timer_to_stop, 2);


while timer_count< timer_to_stop  %(~init_checker(x(1:2,:), goal_condition)) %
tic
    if timer_count == 1
        text(-1.5, -1.5, 'SBC','FontSize',font_size);
        
    end
    handle_timestep = text(-0.5,-3.4,strcat('Time Step = ',num2str(timer_count)),'FontSize',font_size);
    
    x = r.get_poses();
    for i = 1:N
        agent_positions(timer_count, i, :) = x(1:2, i);
    end
   
    
    
    bot_dist_tmp = pdist2(x(1:2,:)', x(1:2,:)');
    bot_dist_tmp = triu(bot_dist_tmp, 1);
    min_bot_dist_tmp = min(bot_dist_tmp(find(bot_dist_tmp(:)>0)));
    min_bot_dist_hist(timer_count) = min_bot_dist_tmp;
    
    min_bot_safe_ratio_tmp = [];
    for i_hehe = 1:(N-1)
            for j_hehe = (i_hehe+1):N
                boxA = [x(1:2,i_hehe)'-([x_rand_span_x(i_hehe);x_rand_span_y(i_hehe)])'-SafetyRadius 2*x_rand_span_x(i_hehe)+2*SafetyRadius 2*x_rand_span_y(i_hehe)+2*SafetyRadius];
                boxB = [x(1:2,j_hehe)'-([x_rand_span_x(j_hehe);x_rand_span_y(j_hehe)])'-SafetyRadius 2*x_rand_span_x(j_hehe)+2*SafetyRadius 2*x_rand_span_y(j_hehe)+2*SafetyRadius];
                area_tmp = rectint(boxA,boxB);
                area_boxA = (2*x_rand_span_x(i_hehe)+2*SafetyRadius)*(2*x_rand_span_y(i_hehe)+2*SafetyRadius);
                area_boxB = (2*x_rand_span_x(j_hehe)+2*SafetyRadius)*(2*x_rand_span_y(j_hehe)+2*SafetyRadius);
                min_bot_safe_ratio_tmp = [min_bot_safe_ratio_tmp; 1-area_tmp/min(area_boxA, area_boxB)];
            end
    end
    
    
    min_bot_safe_ratio_hist(timer_count) = min(min_bot_safe_ratio_tmp);
    
    
    pos_error = r.get_ghost_poses_error();
    
    % add noise to observation

    x_observe = [x(1:2,:)+pos_error; x(3,:)]; % put noise over ground-truth position to get noisy measurements.
    
    
    bot_dist_tmp = pdist2(x_observe(1:2,:)', x_observe(1:2,:)');
    bot_dist_tmp = triu(bot_dist_tmp, 1);
    min_bot_dist_observe_tmp = min(bot_dist_tmp(find(bot_dist_tmp(:)>0)));
    min_bot_dist_observe_hist(timer_count) = min_bot_dist_observe_tmp;
    
    vel_error = (rand(2,N)-0.5).*2;
    vel_error = v_rand_span.*vel_error;
       
    r.set_vel_error(vel_error);
    % finish add noise to motion
    
    if timer_count==2
        r.set_init_bot_true_path(x(1:2,:));
        r.set_init_bot_observe_path(x_observe(1:2,:));
    elseif timer_count>2
        r.set_bot_true_path(x(1:2,:));
        r.set_bot_observe_path(x_observe(1:2,:));
    end
    
   
    dxi = controller(x_observe(1:2, :), goal_condition);
    
    
    dxi_r = si_barrier_certificate(dxi, x_observe(1:2, :));
%     fval_r
    
    delta_dxi = dxi_r-dxi;
    
    pertub_bot_tmp = sum(sum(delta_dxi.^2, 1))/N;
    pertub_bot_hist(timer_count) = pertub_bot_tmp; % 

    dxu = si_to_uni_dynamics(dxi_r, x_observe); 
    dxu_data(timer_count, :) = dxu(:, 1)';
    %disp(dxu);

    r.set_velocities(1:N, dxu);%+vel_error
   
    
    
    
    pos_error = (rand(2,N)-0.5).*2; % get uniformly distributed noise at 2 dimensions for all robots
    pos_error = [x_rand_span_x;x_rand_span_y].*pos_error; %bsxfun(fun_rand,[x_rand_span_x;x_rand_span_y],pos_error); % get the final noise of observation
    
    r.set_ghost_poses_error(pos_error);
    
    r.step();   
   
   
    r.set_video_flag(record_video_flag);
    timer_count = timer_count + 1
    delete(handle_timestep)
 elapsed_time = toc;
 disp(['Elapsed time: ' num2str(elapsed_time) ' seconds']);
end
if record_video_flag
    writerObj = r.get_videoobj();
    close(writerObj);
end
time = 1:timer_to_stop;

% Plotting the graph
%plot(time, dxu_data(:, 1), 'b', time, dxu_data(:, 2), 'r');
%xlabel('Time Step');
%ylabel('Controls');
%title('Graph of dxu with Each Time Step');
%legend('Linear Velocity', 'Angular Velocity');

% Though we didn't save any data, we still should call r.call_at_scripts_end() after our
% experiment is over!
r.call_at_scripts_end();

