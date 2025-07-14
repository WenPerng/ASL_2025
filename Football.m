clear; close all; clc;

%% Simulation Setup
% Ball
Xb = zeros(2,1); % Xb = 6 * rand(2,1) - 3;
% Vb = [5.25;0] - Xb; Vb = 1 * Vb / norm(Vb);
Vb = zeros(2,1); % Vb = 5 * rand(2,1);
ball_kicked = false;

% Team 1 players (right)
N1 = 11;
role_team1 = zeros(1,N1);           % 0: goalkeeper, 1:rear, 2:center, 3:front
X1 = 20*ones(2,N1);
X1(:,1) = [5.25;0];
X1(:,9:11) = [1;0] + [0.5;4] .* rand(2,3) - [0.25;2];
V1 = zeros(2,N1);


% Team 2 players (left)
N2 = 11;
state_team2 = zeros(1,N2);          % 0: goalkeeper, 1:rear, 2:center, 3:front
X2 = 20*ones(2,N2);
X2(:,1) = [-5.25;0];
X2(:,9:11) = -[1;0] - [0.5;4] .* rand(2,3) + [0.25;2];
V2 = zeros(2,N2);


% Parameters --------------------------------------------------------------
% ball
r     = 0.2;            % radius of ball (interaction radius w/ point-size players)
beta  = 0.01;           % distance noise to distance^2 ratio
gamma = 0.5;            % characteristic distance: trust current observation instead of past ones
% goalkeeper
v0    = 0.6;            % normal speed for players (10 m/s)
d_gk  = 1;              % distance for gk to start running
% front players
d_atk = 1;              % distance until attack
d_ang = pi/12;          % error in angle of attack (rad)
d_p2p = 2;              % normal distance between players of same team (formation strategy)
c_r   = 3.4 / 3;        % central strip radius (formation strategy)
atk_ang = 10/180*pi;    % attack formation angle (rad)

% Estimation Variables
Y1   = zeros(2,N1);      % team1's estimation to ball position
Y2   = zeros(2,N2);      % team2's estimation to ball position
U1   = zeros(2,N1);      % team1's estimation to ball velocity direction
U2   = zeros(2,N2);      % team2's estimation to ball velocity direction

% GIF visualization -------------------------------------------------------
fps = 30;
Movie = [];
ball_motion = false;
goal_keeper_estimate = false;

% Time
dt = 1/fps;
T = 0:dt:60;

% Endgame
winner = 'tie';

% Plot initialization -----------------------------------------------------
figure;

%% Function Handle ========================================================
Front_V = @(k,side,Xn,Z,W,Yn) Front_motion(k,side,Xn,Z,W,Yn,r,v0,d_atk,d_ang,d_p2p,c_r,atk_ang);

%% Simulation
for t = 1:length(T)
% Endgame condition =======================================================
    if abs(Xb(2)) < 0.366
        if Xb(1) > 5.25
            winner = 'left'; disp(['The winner is ',winner,'!']);
            break;
        elseif Xb(1) < -5.25
            winner = 'right'; disp(['The winner is ',winner,'!']);
            break;
        end
    end

% Ball: changing the velocity =============================================
    [Vb,ball_kicked] = Ball_velocity([X1,X2],[V1,V2],Xb,Vb,r);
    
% Players: goalkeepers ====================================================
    % Goalkeepers' estimations of the ball --------------------------------
    [Y1(:,1),U1(:,1)] = XV_state_est(Xb,beta,ball_kicked,X1(:,1),Y1(:,1),U1(:,1),gamma,dt);
    [Y2(:,1),U2(:,1)] = XV_state_est(Xb,beta,ball_kicked,X2(:,1),Y2(:,1),U2(:,1),gamma,dt);
    % Target of goalkeepers -----------------------------------------------
    C1 = [5.25-0.55+1.932/2;0]; C2 = -C1;
    R  = 1.932 / sqrt(2);
    V1(:,1) = Goalkeeper_motion(Y1(:,1),U1(:,1),v0,X1(:,1),d_gk,C1,R,'right',r);
    V2(:,1) = Goalkeeper_motion(Y2(:,1),U2(:,1),v0,X2(:,1),d_gk,C2,R,'left',r);
    
% Team State Determination ================================================
    % Estimating the motion of the ball by each player --------------------
    for k = 2:N1
        [Y1(:,k),U1(:,k)] = XV_state_est(Xb,beta,ball_kicked,X1(:,k),Y1(:,k),U1(:,k),gamma,dt);
    end
    for k = 2:N2
        [Y2(:,k),U2(:,k)] = XV_state_est(Xb,beta,ball_kicked,X2(:,k),Y2(:,k),U2(:,k),gamma,dt);
    end
    % Role of players in team ---------------------------------------------
    if true
        role_team1 = {0,2:5,6:8,9:11};
        role_team2 = {0,2:5,6:8,9:11};
    end

% Players: rear ===========================================================


% Players: center =========================================================
    
    
% Players: front ==========================================================
    F1 = cell2mat(role_team1(4));
    F2 = cell2mat(role_team2(4));
    % Actions
    for k = 1:length(F1)
        n = F1(k);
        Z = Group_pos_est(X1(:,n),X1(:,F1),beta);
        W = Group_pos_est(X1(:,n),X2,beta);
        V1(:,n) = Front_V(k,'right',X1(:,n),Z,W,Y1(:,n));
    end
    for k = 1:length(F2)
        n = F2(k);
        Z = Group_pos_est(X2(:,n),X2(:,F2),beta);
        W = Group_pos_est(X2(:,n),X1,beta);
        V2(:,n) = Front_V(k,'left',X2(:,n),Z,W,Y2(:,n));
    end

% Updating the positions ==================================================
    % Ball
    Xb = Xb + dt * Vb;
    Vb = 0.985 * Vb;         % reducing speed due to friction
    ball_kicked = false;
    % Players
    X1 = X1 + dt * V1;
    X2 = X2 + dt * V2;
    
% Plot figure ============================================================= 
    Plot_football_court(T(t));
    Plot_players(Xb,X1,X2,Vb,[ball_motion,goal_keeper_estimate]);

% GIF frame ===============================================================
    Movie = [Movie,getframe(gcf)];
end

%% Generate GIF
for j = 1:length(Movie)
	[image,map]=frame2im(Movie(j));		% converts frame to images
	[im,map2]=rgb2ind(image,256);		% converts RGB images to indexed images
	if j==1
		imwrite(im,map2, 'Football.gif','gif','writeMode','overwrite','delaytime',dt,'loopcount',inf);
	else
		imwrite(im,map2,'Football.gif','gif','writeMode','append','delaytime',dt);
	end
end

%% Functions
% Plotting ================================================================
function Plot_football_court(t)
    plot([0,0],[3.4,-3.4],'black','LineWidth',1);
    field_color = [186,214,197]/255;
    hold on
    rectangle('Position',[-5.25,-3.4,10.5,6.8],'EdgeColor','black','LineWidth',1,'FaceColor',field_color);      % field
    rectangle('Position',[-5.5,-0.366,0.25,0.732],'EdgeColor','black','LineWidth',1,'FaceColor',[1,1,1]);       % goal
    rectangle('Position',[5.25,-0.366,0.25,0.732],'EdgeColor','black','LineWidth',1,'FaceColor',[1,1,1]);       % goal
    rectangle('Position',[-5.25,-0.966,0.55,1.932],'EdgeColor','black','LineWidth',1,'FaceColor',field_color);  % gk zone
    rectangle('Position',[4.7,-0.966,0.55,1.932],'EdgeColor','black','LineWidth',1,'FaceColor',field_color);    % gk zone
    % viscircles([0,0],0.915,'Color','black','LineWidth',1);
    % lines
    theta = -pi:0.01:pi;
    circle_x = 0.915*cos(theta); circle_y = 0.915*sin(theta);
    plot(circle_x,circle_y,'black','LineWidth',1)
    plot([0,0],[3.4,-3.4],'black','LineWidth',1);
    hold off
    axis([-5.5,5.5,-3.41,3.41]);
    axis equal;
    axis off;
    set(gcf,'Color',[1,1,1]);
    subtitle(['$t = ',num2str(t,'%2.2f'),'$'],'Interpreter','latex');
    fontname('Times New Roman');
end
function Plot_players(Xb,X1,X2,Vb,added_details)
    ball_motion = added_details(1);
    goal_keeper_estimate = added_details(2);
    hold on
    scatter(Xb(1,:),Xb(2,:),'white','filled','o','MarkerEdgeColor','black');
    scatter(X1(1,1),X1(2,1),[],[1,0.5,0.5],'filled','o','MarkerEdgeColor','black');
    scatter(X2(1,1),X2(2,1),[],[0.5,0.5,1],'filled','o','MarkerEdgeColor','black');
    scatter(X1(1,2:end),X1(2,2:end),'red','filled','o','MarkerEdgeColor','black');
    scatter(X2(1,2:end),X2(2,2:end),'blue','filled','o','MarkerEdgeColor','black');
    % assisting lines -----------------------------------------------------
    if ball_motion
        plot([Xb(1),Xb(1)+0.5*Vb(1)/norm(Vb)],[Xb(2),Xb(2)+0.5*Vb(2)/norm(Vb)],'Color','white');
    end
    if goal_keeper_estimate
        scatter(Y1(1),Y1(2),'red');
        scatter(Y2(1),Y2(2),'blue');
        plot([Y1(1),Y1(1)+0.5*U1(1)/norm(U1)],[Y1(2),Y1(2)+0.5*U1(2)/norm(U1)],'Color','red');
        plot([Y2(1),Y2(1)+0.5*U2(1)/norm(U2)],[Y2(2),Y2(2)+0.5*U2(2)/norm(U2)],'Color','blue');
    end
    hold off
end

% Ball Motion =============================================================
function [Vb,ball_kicked] = Ball_velocity(X,V,Xb,Vb,r)
    % kick / push by players ----------------------------------------------
    ball_kicked = false;
    dist_b2p = vecnorm(X - Xb);
    normal = (-X + Xb) ./ vecnorm(-X + Xb);
    dV = V - Vb;
    dV_kick = zeros(2,1);
    for n = 1:size(X,2)
        if dist_b2p(n)<r
            disp('kick');
            ball_kicked = true;
            dV_kick = dV_kick + (dV(:,n)' * normal(:,n)) * normal(:,n);
        end
    end
    Vb = Vb + 2 * dV_kick; % players assumed to have mass way larger than ball
    % hitting a wall ------------------------------------------------------
    if Xb(1) > 5.25 - 0.5 * r
        if abs(Xb(2)) > 0.366 && Vb(1) > 0
            Vb(1) = -Vb(1);
        end
    elseif -Xb(1) > 5.25 - 0.5 * r
        if abs(Xb(2)) > 0.366 && Vb(1) < 0
            Vb(1) = -Vb(1);
        end
    elseif Xb(2) > 3.4 - 0.5 * r && Vb(2) > 0
        Vb(2) = -Vb(2);
    elseif -Xb(2) > 3.4 - 0.5 * r && Vb(2) < 0
        Vb(2) = -Vb(2);
    end
end

% State Estimation ========================================================
function [Y,U] = XV_state_est(Xb,beta,ball_kicked,X,Y,U,gamma,dt)
    dist_b2gk = norm(Xb-X);
    Y_old = Y;
    Y = Xb + beta * dist_b2gk * randn(2,1);
    if ball_kicked
        U = (Y-Y_old)/dt;
    else
        mu = exp(-norm(Y - X) / gamma);
        U = (1-mu) * (Y-Y_old)/dt + mu * U;
    end
end
function Z = Group_pos_est(X,X2,beta)
    % Estimate the location of enemy team memebers
    dist_p2p = norm(X2 - X);
    Z = X2 + beta * dist_p2p * rand(size(X2));
end

% Player: Goalkeeper Motion ===============================================
function V = Goalkeeper_motion(Y,U,v0,X,d_gk,Circle,Radius,side,r)
    if norm(Y-X) > d_gk
        T = Y + ((X - Y)' * U) * U / (U' * U);
        switch lower(side)
            case 'right'
                if norm(T - Circle) > Radius || T(1) > 5.25
                    T = Circle + Radius * (Y - Circle) / norm(Y - Circle);
                end
            case 'left'
                if norm(T - Circle) > Radius || T(1) < -5.25
                    T = Circle + Radius * (Y - Circle) / norm(Y - Circle);
                end
        end
    else
        switch lower(side)
            case 'right'
                if norm(Y-X) > 1.5*norm([5.25;0]-X)
                    T = [5.25-0.5*r;Y(2)];
                else
                    T = Y;
                end
            case 'left'
                if norm(Y-X) > 1.5*norm([-5.25;0]-X)
                    T = [-5.25+0.5*r;Y(2)];
                else
                    T = Y;
                end
        end
        
    end
    V = v0 * (T - X) / norm(T - X);
    switch lower(side)
        case 'right'
            if norm(Y-[5.25;0]) > d_gk
                V = 0.5 * V;
            end
        case 'left'
            if norm(Y-[-5.25;0]) > d_gk
                V = 0.5 * V;
            end
    end
end

% Player: General Motion ==================================================
function V = Kick_ball(X,Xb,d_v,v0,r,d_atk,d_ang)
% Player at position 'X' wants to kick ball at 'Xb' in the direction 'd_v'.
% The function outputs the velocity required for the player.
    d_p2b = X - Xb;
    dist_b2p = norm(d_p2b);
    if dist_b2p > d_atk
        V = v0 * (Xb-X) / norm(Xb-X);
    else
        if d_v' * d_p2b > 0
            N = d_p2b - (d_p2b' * d_v) * d_v / (d_v' * d_v);
            Target = Xb + 1.5*r * N / norm(N);
        elseif d_v' * d_p2b / norm(d_v) / norm(d_p2b) > cos(pi-d_ang)
            Target = Xb - 1.5*r * d_v / norm(d_v);
        else
            Target = Xb - 0.8*r * d_v / norm(d_v);
        end
        V = v0 * (Target-X) / norm(Target-X);
    end
end

% Plyaer: Front Motion ====================================================
function V = Front_forward(k,side,X,Z,W,Xb,r,v0,d_atk,d_ang,d_p2p,c_r,atk_ang)
    switch lower(side)
        case 'right'
            goal = [-5.25;0];
        case 'left'
            goal = [5.25;0];
    end
    s = sign(goal(1));

    % Actions (to kick ball or to move forward) ===========================
    % either be main ball-bearer, or are cooperators
    dist_b2p = vecnorm(Xb - Z);
    [~,closest_k] = min(dist_b2p);
    if closest_k == k
        % Main ball-bearer ------------------------------------------------
        % get close to the ball and pass it forward
        % while avoiding enemy agents
        d_v = goal / norm(goal);  % intended direction for ball velocity
        [dist,index] = min(vecnorm(W - X));
        if dist < 3*r && d_v' * (W(:,index) - Xb) > 0
            N = W(:,index) - Xb; N = [-N(2);N(1)];
            if N' * d_v < 0
                N = -N;
            end
            d_v = d_v + N/norm(N);
        end
        V = Kick_ball(X,Xb,d_v,v0,r,d_atk,d_ang);
    else
        % Cooporating -----------------------------------------------------
        % to move in an uniform line under 4-3-3 formation
        if size(Z,2) == 3
            y = Z(2,:);
            y_k = y(k);
            y(k) = 10; y(closest_k) = 10;
            y_other = min(y);
            if y_k > y_other
                position = 'above';
            else
                position = 'below';
            end
        else
            position = 'null';
            disp('Only the 4-3-3 formation is valid now, others are still WIP.');
        end
        if abs(Z(2,closest_k)) < c_r
            % ball-bearer at center
            switch position
                case 'above'
                    Target = Z(:,closest_k) + d_p2p * [s*tan(atk_ang);1];
                case 'below'
                    Target = Z(:,closest_k) + d_p2p * [s*tan(atk_ang);-1];
            end
        elseif Z(2,closest_k) > 0
            % ball-bearer at top edge
            switch position
                case 'above'
                    Target = Z(:,closest_k) - d_p2p * [s*tan(atk_ang);1];
                case 'below'
                    Target = Z(:,closest_k) - d_p2p * [0;2];
            end
        else
            % ball-bearer at bottom edge
            switch position
                case 'above'
                    Target = Z(:,closest_k) + d_p2p * [0;2];
                case 'below'
                    Target = Z(:,closest_k) + d_p2p * [-s*tan(atk_ang);1];
            end
        end
        V = v0 * (Target-X) / norm(Target-X);
    end
end
function V = Front_attack(k,side,X,Z,W,Xb,r,v0,d_atk,d_ang,d_p2p,c_r,atk_ang)
    switch lower(side)
        case 'right'
            goal = [-5.25;0];
        case 'left'
            goal = [5.25;0];
    end
    s = sign(goal(1));
    

    % Actions (to score goal or pass ball) ================================
    % either be main ball-bearer, or are cooperators
    dist_b2p = vecnorm(Xb - Z);
    [~,closest_k] = min(dist_b2p);
    if closest_k == k
        % Main ball-bearer ------------------------------------------------
        % get close to the ball and pass it forward
        % while avoiding enemy agents
        d_v = goal - Xb; d_v = d_v / norm(d_v); % intended direction for ball velocity
        W_ang = d_v' * (W-Xb) ./ vecnorm(W-Xb);
        if max(W_ang) > cos(atk_ang)
            % line of sight is NOT clear, pass the ball
            Target = ;
            d_v = Target - Xb;
        end
        V = Kick_ball(X,Xb,d_v,v0,r,d_atk,d_ang);
    else
        % Cooporating -----------------------------------------------------
        % keep line of sight to ball-bearer
        % get the ball passed to him
    end
end
function V = Front_motion(k,side,X,Z,W,Xb,r,v0,d_atk,d_ang,d_p2p,c_r,atk_ang)
    % Determine if one should attack or move forward with the ball
    % state -- 0:idle, 1:go forward, 2:attack
    switch lower(side)
        case 'right'
            goal = [-5.25;0];
        case 'left'
            goal = [5.25;0];
    end
    
    % Motion ==============================================================
    if abs(Y(1)) > 5.25 / 3 && abs(X(1)) > 5.25 /3
        % attack!
        V = Front_attack(k,side,X,Z,W,Xb,r,v0,d_atk,d_ang,d_p2p,c_r,atk_ang);
    elseif abs(Y(1) - goal(1)) < 4
        % go forward
        V = Front_forward(k,side,X,Z,W,Xb,r,v0,d_atk,d_ang,d_p2p,c_r,atk_ang);
    else
        % idle
        V = zeros(2,1); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end



