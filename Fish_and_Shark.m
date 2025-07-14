% This code generates a simulation of schools of fish chasing food
% and with sharks chasing the fish.
% The view is contained inside a box of size 1x1.
clear; close all; clc;

%% Simulation Setup
% States for fish
K1 = 1;
X = 0.4 * rand(2,K1);
v0 = 0.3;                   % speed limit for fish
V = 2*v0 * rand(2,K1) - v0;

% Food
K2 = 1;
F = [0.5;0.5] + 0.5 * rand(2,K2);
rho = 0.03;

% States for shark
K3 = 3;
Y = (0.1*rand(2,K3) - 0.05) + [0.5;0.5];
u0 = 0.5;                   % speed limit for sharks
U = 2*u0 * rand(2,K3) - u0;
state_shark = zeros(1,K3);
Te_accu = zeros(1,K3);      % accumulated encircling time

% Behavioral parameters
lambda = 0.2;                           % inertia
beta   = 0.03;  Vb = zeros(size(V));    % group up
gamma  = 0.3;   Vg = zeros(size(V));    % move with CoG
mu     = 0.7;                           % loss factor for move w/ CoG
delta  = 10;    Vd = zeros(size(V));    % fish collision avoidance
eps    = 1;     Ve = zeros(size(V));    % attract by food
sigma  = 0.7;   Vs = zeros(size(V));    % avoiding sharks (tradeoff w/ Ve)
alpha  = 1;     Va = zeros(size(V));    % away from boundary
r      = 0.1;                           % min distance between fish
rs    = 0.08;                           % awareness distance betw/ f & s
rc    = 0.2;                            % chase radius of shark
re    = 0.15;                            % encircling radius of shark
Te    = 2;                              % encircling time limit
Delta = 2;      Ud = zeros(size(U));    % shark collision avoidance
Sigma = 1;      Us = zeros(size(U));    % shark chasing fish

% Wall regions
theta = 0:0.001:2*pi;
pillar_r = 0.1;
pillar = [0.5+pillar_r*cos(theta);0.5+pillar_r*sin(theta)];
Potential = @(X) norm(X-0.5)-pillar_r;

% GIF visualization
fps = 20;
Movie = [];
include_adjacency = true;

% Time
dt = 1/fps;
T = 0:dt:10;

% Plot initialization -----------------------------------------------------
tail_length = 0.03;
s_tail_length = 0.04;
figure;
plot([X(1,:);X(1,:)-tail_length*V(1,:)./vecnorm(V)],[X(2,:);X(2,:)-tail_length*V(2,:)./vecnorm(V)],'black');
hold on
scatter(X(1,:),X(2,:),'yellow','filled','o','MarkerEdgeColor','black');
scatter(F(1,:),F(2,:),'red','filled','o');
plot([Y(1,:);Y(1,:)-s_tail_length*U(1,:)./vecnorm(U)],[Y(2,:);Y(2,:)-s_tail_length*U(2,:)./vecnorm(U)],'blue','LineWidth',1);
scatter(Y(1,:),Y(2,:),'blue','filled','o');
fill(pillar(1,:),pillar(2,:),[0.8,0.8,1],'EdgeColor','black','FaceAlpha',0.5);
hold off
axis([-0.1,1.1,-0.1,1.1]);
axis square;
set(gcf,'Color',[1,1,1]);
Movie = [Movie,getframe(gcf)];

%% Simulation
for t = 1:length(T)
% Food dynamics ===========================================================
    % food
    for f = 1:K2
        dist_f2f = vecnorm(X - F(:,f));
        if min(dist_f2f) < rho
            F(:,f) = rand(2,1); % replenish new food once one is eaten
        end
    end
    % fish
    eaten = zeros(1,K1);
    for k = 1:K1
        dist_f2s = vecnorm(Y - X(:,k));
        if min(dist_f2s) < rho
            eaten(k) = 1;
        end
    end
    for k = K1:-1:1
        if eaten(k) == 1
            X(:,k) = [];
            V(:,k) = [];
            Vb(:,k) = []; Vg(:,k) = []; Vd(:,k) = []; Ve(:,k) = []; Vs(:,k) = []; Va(:,k) = [];
        end
    end
    K1 = K1 - sum(eaten);

% Fish dynamics ===========================================================
    % 0. Find group (adjacency matrix & combination matrix) ---------------
    Adjacency = zeros(K1,K1);
    for k = 1:K1
        dist_f2f = vecnorm(X - X(:,k));
        Adjacency(k,:) = dist_f2f < 1.5 * r;
    end
    [Combination,~] = generate_combination_policy(Adjacency,0,'metropolis');
    % 1. Grouping up ------------------------------------------------------
    for k = 1:K1
        if sum(Adjacency(:,k)) == 1
            Vb(:,k) = zeros(2,1);
        else
            dist_f2f = vecnorm(X - X(:,k));
            dist_f2f(k) = inf;
            [~,j] = min(dist_f2f);
            Vb(:,k) = (X(:,j) - X(:,k)) / norm(X(:,j) - X(:,k));
        end
    end
    % 2. Move with center of gravity --------------------------------------
    Vg = ((1-mu) * Vg + mu * V) * Combination';
    % 3. Avoid collision --------------------------------------------------
    Vd = zeros(2,K1);
    for k = 1:K1
        if sum(Adjacency(:,k)) == 1
            continue
        end
        for j = 1:K1
            if j == k
                continue
            end
            if Adjacency(j,k)
                d_f2f = X(:,j) - X(:,k);
                Vd(:,k) = Vd(:,k) + (norm(d_f2f) - r) * d_f2f / norm(d_f2f);
            end
        end
        % Vd(:,k) = Vd(:,k) / (sum(Adjacency(:,k))-1);
    end
    % 4. Attracted by closest food & avoiding sharks ----------------------
    % 4-1. find state
    closest_s2f = zeros(1,K1);
    for k = 1:K1
        dist_f2s = vecnorm(Y - X(:,k));
        [~,s] = min(dist_f2s);
        closest_s2f(k) = s;
    end
    state_fish = zeros(1,K1);
    for k = 1:K1
        s = closest_s2f(k);
        d_f2s = Y(:,s) - X(:,k);
        if norm(d_f2s) > 2*rs
            state_fish(k) = 1;          % region I (outside)
        elseif norm(d_f2s) < rs
            state_fish(k) = 4;          % region IV (inside)
        else
            if d_f2s' * U(:,s) < 0
                state_fish(k) = 2;      % region II (front)
            else
                state_fish(k) = 3;      % region III (rear)
            end
        end
    end
    % 4-2. actions based on region
    for k = 1:K1
        s = closest_s2f(k);
        switch state_fish(k)
            case 1
                dist_f2f = vecnorm(F - X(:,k));
                [~,f] = min(dist_f2f);
                Ve(:,k) = (F(:,f) - X(:,k)) / norm(F(:,f) - X(:,k));
            case 2
                temp_V = (X(:,k) - Y(:,s)) / norm(X(:,k) - Y(:,s));
                Ve(:,k) = 2 * [-temp_V(2);temp_V(1)];
                if temp_V' * [-U(2,s);U(1,s)] < 0
                    Ve(:,k) = - Ve(:,k);
                end
                Ve(:,k) = Ve(:,k) + temp_V;
            case 3
                Ve(:,k) = - U(:,s)/norm(U(:,s)) + (X(:,k)-Y(:,s)) / norm(X(:,k)-Y(:,s));
            case 4
                Ve(:,k) = (2*rs/norm(X(:,k)-Y(:,s)) - 1) * ...
                            (X(:,k)-Y(:,s)) / norm(X(:,k)-Y(:,s));
        end
    end
    % 5. Away from boundary -----------------------------------------------
    Va = zeros(2,K1);
    for k = 1:K1
        if X(1,k) > 1
            Va(1,k) = -1;
        elseif X(1,k) < 0
            Va(1,k) = 1;
        end
        if X(2,k) > 1
            Va(2,k) = -1;
        elseif X(2,k) < 0
            Va(2,k) = 1;
        end
        if Potential(X(:,k)) < 0.05
            n = (X(:,k) - [0.5;0.5]) / norm(X(:,k) - [0.5;0.5]);
            n_perp = [-n(2);n(1)];
            if V(:,k)' * n_perp > 0
                Va(:,k) = 0.5 * n + n_perp;
            else
                Va(:,k) = 0.5 * n - n_perp;
            end
        end
    end
    % 7. Changing the velocity --------------------------------------------
    V = lambda * V + (1-lambda) * v0 * (beta * Vb + gamma * Vg + delta * Vd ...
                                        + eps * Ve + alpha * Va);
    for k = 1:K1
        if state_fish(k) == 1
            if norm(V(:,k)) > v0
                V(:,k) = v0 * V(:,k) / norm(V(:,k));
            end
        else
            if norm(V(:,k)) > 2*v0
                V(:,k) = 2*v0 * V(:,k) / norm(V(:,k));
            end
        end
    end

% Shark Dynamics ==========================================================
    % 0. Find target (group and fish) -------------------------------------
    % -- Network center for fish
    Z = zeros(2,K1);
    for k = 1:K1
        Z(:,k) = X * Adjacency(:,k) / sum(Adjacency(:,k));
    end
    % -- Closest fish to shark
    closest_fg2s = zeros(1,K3); % (fg = fish group)
    closest_f2s = zeros(1,K3);
    for s = 1:K3
        dist_fg2s = vecnorm(Z - Y(:,s));
        dist_f2s = vecnorm(X - Y(:,s));
        [~,g] = min(dist_fg2s);
        [~,k] = min(dist_f2s);
        closest_fg2s(s) = g;
        closest_f2s(s) = k;
    end
    % 1. State determination ----------------------------------------------
    % -- Adjacency matrix for sharks & persuading
    Adjacency_shark = zeros(K3,K3);
    for s = 1:K3
        dist_s2s = vecnorm(Y - Y(:,s));
        Adjacency_shark(s,:) = dist_s2s < 3 * r;
    end
    dist_fg2s = vecnorm(Z(:,closest_fg2s) - Y);
    for s = 1:K3
        neighbor = (Adjacency_shark(:,s)==1);
        neighbor_dist_fg2s = dist_fg2s(neighbor);
        neighbor_closest_fg = closest_fg2s(neighbor);
        [~,neighbor] = min(neighbor_dist_fg2s);
        closest_fg2s(s) = neighbor_closest_fg(neighbor);
    end
    Adjacency_shark = (closest_fg2s' == closest_fg2s); 
    % -- state determination
    for s = 1:K3
        if state_shark(s) == 3
            continue
        end
        g = closest_fg2s(s);
        if norm(Z(:,g) - Y(:,s)) > rc
            state_shark(s) = 0;
        % elseif norm(X(:,k)-Y(:,s)) < 0.8*rc && norm(X(:,k)-Z(:,k)) > re
        %     state_shark(s) = 2;
        else
            state_shark(s) = 1;
        end
    end
    
    % 2. Action based on state --------------------------------------------
    for s = 1:K3
        if Te_accu(s) > Te                      % signal for attack
            disp('Atk!'); %=======================================================
            Te_accu(Adjacency_shark(:,s)==1) = 0;
            neighbor = 1:K3; neighbor = neighbor(Adjacency_shark(:,s)==1);
            rand_index = randperm(length(neighbor));
            state_shark(neighbor(rand_index(1))) = 3;
        end
    end
    for s = 1:K3
        g = closest_fg2s(s);
        k = closest_f2s(s);
        disp(state_shark(1));
        switch state_shark(s)
            case 0 % chase
                Us(:,s) = (Z(:,g) - Y(:,s)) / norm(Z(:,g) - Y(:,s));
            case 1 % encircle
                Te_accu(s) = Te_accu(s) + dt;
                U(:,s) = (Y(:,s) - Z(:,g)) / norm(Y(:,s) - Z(:,g));
                U(:,s) = [-U(2,s);U(1,s)];
                % neighbor_positions = Y(:,Adjacency_shark(:,s)==1);
                % [~,closest_neighbor] = min(vecnorm(neighbor_positions - Y(:,s)));
                % neighbor_direction = neighbor_positions(closest_neighbor) - Z(:,g);
                neighbor_position = sum(Y(:,Adjacency_shark(:,s)==1),2);
                neighbor_position = neighbor_position - Y(:,s);
                neighbor_position = neighbor_position / (sum(Adjacency_shark(:,s))-1);
                neighbor_direction = neighbor_position - Z(:,g);
                n = [-neighbor_direction(2);neighbor_direction(1)];
                n = n / norm(n);
                if U(:,s)' * n < 0
                    U(:,s) = - U(:,s);
                end
                if norm(Y(:,s)-X(:,k)) < re
                    U(:,s) = U(:,s) + (Y(:,s)-X(:,k)) / norm(Y(:,s)-X(:,k));
                end
            case 2 % trap
                Te_accu(s) = Te_accu(s) + dt;
                disp('WIP') %==================================================
            case 3 % attack
                %==============================================================
                state_shark(s) = 1; % temp (WIP)
        end
    end
    % 3. Avoid collision --------------------------------------------------
    Ud = zeros(2,K3);
    for s = 1:K3
        if sum(Adjacency_shark(:,s)) == 1
            continue
        end
        for j = 1:K3
            if j == s
                continue
            end
            if Adjacency_shark(j,s)
                d_s2s = Y(:,j) - Y(:,s);
                Ud(:,s) = Ud(:,s) + (norm(d_s2s) - r) * d_s2s / norm(d_s2s);
            end
        end
    end
    % 4. Away from boundary -----------------------------------------------
    Ua = zeros(2,K3);
    for s = 1:K3
        % if X(1,k) > 1
        %     Va(1,k) = -1;
        % elseif X(1,k) < 0
        %     Va(1,k) = 1;
        % end
        % if X(2,k) > 1
        %     Va(2,k) = -1;
        % elseif X(2,k) < 0
        %     Va(2,k) = 1;
        % end
        if Potential(Y(:,s)) < 0.05
            n = (Y(:,s) - [0.5;0.5]) / norm(Y(:,s) - [0.5;0.5]);
            n_perp = [-n(2);n(1)];
            if U(:,s)' * n_perp > 0
                Ua(:,s) = 0.5 * n + n_perp;
            else
                Ua(:,s) = 0.5 * n - n_perp;
            end
        end
    end
    % 5. Changing the velocity --------------------------------------------
    U = 0.1 * U + 0.9 * u0 * (Sigma * Us + Delta * Ud + alpha * Ua);
    for s = 1:K3
        if state_shark(s) == 3
            if norm(U(:,s)) > 1.2*u0
                U(:,s) = 1.2*u0 * U(:,s) / norm(U(:,s));
            end
        else
            if norm(U(:,s)) > v0
                U(:,s) = u0 * U(:,s) / norm(U(:,s));
            end
        end
    end

% Changing the positions ==================================================
    % Fish
    % X = X + dt * V;
    % Shark
    Y = Y + dt * U;
    
% Plot figure =============================================================   
    plot([X(1,:);X(1,:)-tail_length*V(1,:)./vecnorm(V)],[X(2,:);X(2,:)-tail_length*V(2,:)./vecnorm(V)],'black');
    
    if include_adjacency
        hold on
        for k = 1:K1
            for j = 1:K1
                if Adjacency(k,j)==1
                    plot([X(1,k),X(1,j)],[X(2,k),X(2,j)],'LineWidth',0.1,'Color',[0.8,0.8,0.8]);
                end
            end
        end
        for k = 1:K3
            for j = 1:K3
                if Adjacency_shark(k,j)==1
                    plot([Y(1,k),Y(1,j)],[Y(2,k),Y(2,j)],'LineWidth',0.1,'Color',[0.8,0.8,0.8]);
                end
            end
            plot([Y(1,k),Z(1,closest_fg2s(k))],[Y(2,k),Z(2,closest_fg2s(k))],'LineWidth',0.1,'Color',[0.2,0,0]);
        end
        hold off
    end
    
    hold on
    scatter(X(1,:),X(2,:),'yellow','filled','o','MarkerEdgeColor','black');
    scatter(F(1,:),F(2,:),'red','filled','o');
    plot([Y(1,:);Y(1,:)-s_tail_length*U(1,:)./vecnorm(U)],[Y(2,:);Y(2,:)-s_tail_length*U(2,:)./vecnorm(U)],'blue','LineWidth',1);
    scatter(Y(1,:),Y(2,:),'blue','filled','o');
    fill(pillar(1,:),pillar(2,:),[0.8,0.8,1],'EdgeColor','black','FaceAlpha',0.5);
    hold off
    axis([-0.1,1.1,-0.1,1.1]);
    axis square;

% GIF frame ===============================================================
    Movie = [Movie,getframe(gcf)];
end

%% Generate GIF
for j = 1:length(Movie)
	[image,map]=frame2im(Movie(j));		% converts frame to images
	[im,map2]=rgb2ind(image,256);		% converts RGB images to indexed images
	if j==1
		imwrite(im,map2, 'Fish_and_Shark.gif','gif','writeMode','overwrite','delaytime',dt,'loopcount',inf);
	else
		imwrite(im,map2,'Fish_and_Shark.gif','gif','writeMode','append','delaytime',dt);
	end
end