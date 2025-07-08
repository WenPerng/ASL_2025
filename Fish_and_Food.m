% This code generates a simulation of schools of fish chasing food.
% The fish is contained inside a box of size 1x1.
clear; close all; clc;

%% Simulation Setup
% States for fish
K1 = 10;
X = rand(2,K1);
v0 = 0.3;                   % speed limit for fish
V = 2*v0 * rand(2,K1)-v0;

% Food
K2 = 1;
F = rand(2,K2);
rho = 0.03;

% Behavioral parameters
lambda = 0.2;                           % inertia
beta   = 0.03;     Vb = zeros(size(V));    % group up
gamma  = 0.3;   Vg = zeros(size(V));    % move with CoG
mu     = 0.7;                           % loss factor for move w/ CoG
delta  = 10;    Vd = zeros(size(V));    % collision avoidance
eps    = 1;     Ve = zeros(size(V));    % attract by food
alpha  = 1;     Va = zeros(size(V));    % away from boundary
r      = 0.1;                           % min distance between fish

% Wall regions
theta = 0:0.001:2*pi;
pillar = [0.5+0.1*cos(theta);0.5+0.1*sin(theta)];
Potential = @(X) norm(X-0.5)-0.1;
for k = 1:K1
    while Potential(X(:,k)) < 0.05
        X(:,k) = rand(2,1);
    end
end
for f = 1:K2
    while Potential(F(:,f)) < 0.05
        F(:,f) = rand(2,1);
    end
end

% GIF
fps = 20;
Movie = [];

% Time
dt = 1/fps;
T = 0:dt:10;

% Plot initialization -----------------------------------------------------
figure;
tail_length = 0.15;
plot([X(1,:);X(1,:)-tail_length*V(1,:)],[X(2,:);X(2,:)-tail_length*V(2,:)],'black');
hold on
scatter(X(1,:),X(2,:),'yellow','filled','o','MarkerEdgeColor','black');
scatter(F(1,:),F(2,:),'red','filled','o');
fill(pillar(1,:),pillar(2,:),[0.8,0.8,1],'EdgeColor','black','FaceAlpha',0.5);
hold off
axis([-0.1,1.1,-0.1,1.1]);
axis square;
set(gcf,'Color',[1,1,1]);
Movie = [Movie,getframe(gcf)];

%% Simulation
for t = 1:length(T)
% Food dynamics =======================================================
    for f = 1:K2
        dist_f2f = vecnorm(X - F(:,f));
        if min(dist_f2f) < rho
            F(:,f) = rand(2,1); % replenish new food once one is eaten
            while Potential(F(:,f)) < 0.05
                F(:,f) = rand(2,1);
            end
        end
    end

% Fish dynamics =======================================================
    % 0. Find group (adjacency matrix & combination matrix)
    Adjacency = zeros(K1,K1);
    for k = 1:K1
        dist_f2f = vecnorm(X - X(:,k));
        Adjacency(k,:) = dist_f2f < 1.5 * r;
    end
    [Combination,~] = generate_combination_policy(Adjacency,0,'metropolis');
    % 1. Grouping up
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
    % 2. Move with center of gravity
    Vg = ((1-mu) * Vg + mu * V) * Combination';
    % 3. Avoid collision
    Vd = zeros(size(Vd));
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
    % 4. Attracted by closest food
    for k = 1:K1
        dist_f2f = vecnorm(F - X(:,k));
        [~,f] = min(dist_f2f);
        Ve(:,k) = (F(:,f) - X(:,k)) / norm(F(:,f) - X(:,k));
    end
    % 5. Away from boundary
    Va = zeros(size(Va));
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
    % 6. Changing the velocity --------------------------------------------
    V = lambda * V + (1-lambda) * v0 * (beta * Vb + gamma * Vg + delta * Vd ...
                                   + eps * Ve + alpha * Va);
    for k = 1:K1
        if norm(V(:,k)) > v0
            V(:,k) = v0 * V(:,k) / norm(V(:,k));
        end
    end
    % 7. Changing the position --------------------------------------------
    X = X + dt * V;
    
% Plot figure =========================================================
    plot([X(1,:);X(1,:)-tail_length*V(1,:)],[X(2,:);X(2,:)-tail_length*V(2,:)],'black');
    hold on
    scatter(X(1,:),X(2,:),'yellow','filled','o','MarkerEdgeColor','black');
    scatter(F(1,:),F(2,:),'red','filled','o');
    fill(pillar(1,:),pillar(2,:),[0.8,0.8,1],'EdgeColor','black','FaceAlpha',0.5);
    hold off
    axis([-0.1,1.1,-0.1,1.1]);
    axis square;

% GIF frame ===========================================================
    Movie = [Movie,getframe(gcf)];
end

%% Generate GIF
for j = 1:length(Movie)
	[image,map]=frame2im(Movie(j));		% converts frame to images
	[im,map2]=rgb2ind(image,256);		% converts RGB images to indexed images
	if j==1
		imwrite(im,map2, 'Fish_and_Food.gif','gif','writeMode','overwrite','delaytime',dt,'loopcount',inf);
	else
		imwrite(im,map2,'Fish_and_Food.gif','gif','writeMode','append','delaytime',dt);
	end
end