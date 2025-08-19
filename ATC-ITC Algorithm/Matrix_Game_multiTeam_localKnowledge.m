%% ATC-ITC over Games between a Network of Mutiple Teams
% This code does not require any further toolbox to run.
%
% This code simulates the performance of the ATC-ITC (adapt-then-combine,
% inference-then-combin) algorithm.
% The step-sizes are diminishing in the rate of 1/i.
clear; close all; clc;

%% Setting Up the Simulation: Variables, Parameters, and Function Handles
cell2Mat = @(cellOfMatrices, index) cellToMatrixConversion(cellOfMatrices, index);

progressBarType = 2; % 1, 2

% Descent Parameters ------------------------------------------------------
stepsize    = 0.001 * 2 .^ (-5 : -1 : -5);          % all step-sizes to be tested
totalIter   = 1 * 10 ^ 4;                           % number of iterations
totalSample = 010;                                  % number of samples

% Noise
sigma = 1.00;           % radius of the uniformly-dirstributed gradient noise

% Convexity
lambda = 050;           % ratio between epsilon and delta (value > T-1)!

% Game Parameters ---------------------------------------------------------
Kt = [3, 3, 3];         % number of players in each team
K  = sum(Kt);           % total number of players
Mt = [1, 1, 2];         % strategy size of each team
M  = sum(Mt);           % total dimension of strategies
T  = length(Kt);        % total number of teams

% Block Sizes
playerLowerLim = @(tTeam) sum(Kt(1 : tTeam - 1)) + 1;
playerUpperLim = @(tTeam) sum(Kt(1 : tTeam));
stratLowerLim  = @(tTeam) sum(Mt(1 : tTeam - 1)) + 1;
stratUpperLim  = @(tTeam) sum(Mt(1 : tTeam));
blkLowerLim    = @(tau, t) K * sum(Mt(1 : tau - 1)) + sum(Kt(1 : t - 1)) * Mt(tau) + 1;
blkUpperLim    = @(tau, t) K * sum(Mt(1 : tau - 1)) + sum(Kt(1 : t)) * Mt(tau);

% Network Structure -------------------------------------------------------
genMultiTeamNetwork(Kt);
load("Network_structure_multiTeam.mat");

% Perron Vector of Each Team
p = cell(T, 1);
for tTeam = 1 : T
    [V, E]     = eig(cell2Mat(A, tTeam));
    [~, index] = max(diag(E));
    pt         = V(:, index);
    pt         = pt / sum(pt);
    p(tTeam)   = {pt};
end

% Combination Matrix of Each Team
B = cell(T, 1);
for sTeam = 1 : T
    tempB = zeros(K, K);
    for tTeam = 1 : T
        lowerLim = playerLowerLim(tTeam);
        upperLim = playerUpperLim(tTeam);
        if tTeam == sTeam
            blkRow = [];
            for kPlayer = 1 : T
                blkRow = [blkRow, cell2Mat(C, [sTeam, kPlayer])];
            end
            tempB(lowerLim : upperLim, :) = cell2Mat(A, tTeam) * blkRow;
            tempB(lowerLim : upperLim, lowerLim : upperLim) = cell2Mat(A, tTeam);
        else
            tempB(lowerLim : upperLim, lowerLim : upperLim) = cell2Mat(C, [tTeam, tTeam]);
        end
    end
    B(sTeam) = {tempB};
end

% Local Cost Functions ----------------------------------------------------
genMultiTeamMatrixGame(Kt, Mt, lambda);
load("Matrix_game_data_multiTeam.mat"); 

% Gradient Function Handles
localGradient  = @(tTeam, kPlayer, x) computeLocalGradient(tTeam, kPlayer, x, Mt, Ak, bk, Ck);
globalGradient = @(x) computeGlobalGradient(p, x, Kt, Mt, Ak, bk, Ck);

% Satistics ---------------------------------------------------------------
vecX2NE  = zeros(K * M, totalIter, totalSample);
vecY2NE  = zeros(K * M, totalIter, totalSample);
distX2NE = zeros(1, length(stepsize));
convRate = zeros(1, length(stepsize));

% Plot Set Up -------------------------------------------------------------
figure;
set(gcf, 'Color', [1,1,1]);

%% Nash Equilibrium: Calculating the NE of the Game Specified
AkAgg = cell(T, 1);
bkAgg = cell(T, 1);
CkAgg = cell(T, 1);

% Aggregating Each Team's Loss Functions
for tTeam = 1 : T
    AkAggTemp  = 0;
    bkAggTemp  = 0;
    CkAggTemp  = 0;
    pt         = cell2Mat(p, tTeam);
    lowerRange = playerLowerLim(tTeam) - 1;
    for kPlayer = 1 : Kt(tTeam)
        AkAggTemp = AkAggTemp + pt(kPlayer) * cell2Mat(Ak, lowerRange + kPlayer);
        bkAggTemp = bkAggTemp + pt(kPlayer) * cell2Mat(bk, lowerRange + kPlayer);
        CkAggTemp = CkAggTemp + pt(kPlayer) * cell2Mat(Ck, lowerRange + kPlayer);
    end
    AkAgg(tTeam) = {AkAggTemp};
    bkAgg(tTeam) = {bkAggTemp};
    CkAgg(tTeam) = {CkAggTemp}; 
end

% computing Nash equlibrium
NashMat = zeros(M, M);
NashVec = zeros(M, 1);
for tTeam = 1 : T
    lowerLim = stratLowerLim(tTeam);
    upperLim = stratUpperLim(tTeam);

    NashMat(lowerLim : upperLim, :)                   = cell2Mat(CkAgg, tTeam);
    NashMat(lowerLim : upperLim, lowerLim : upperLim) = cell2Mat(AkAgg, tTeam);
    NashVec(lowerLim : upperLim)                      = cell2Mat(bkAgg, tTeam);
end

xStar = - inv(NashMat) * NashVec;
XStar = zeros(K * M, 1);
for tTeam = 1 : T
    XStar(K * sum(Mt(1 : tTeam - 1)) + 1 : K * sum(Mt(1 : tTeam))) = ...
        kron(ones(K, 1), xStar(sum(Mt(1 : tTeam - 1)) + 1 : sum(Mt(1 : tTeam))));
end

%% Recursion: Running the ATC-ITC Algorithm
for nStepsize = 1 : length(stepsize)
    mu = stepsize(nStepsize);
    for nSample = 1 : totalSample
        if progressBarType == 1
            disp(['Progress:', num2str(nStepsize), '/', num2str(length(stepsize)), ...
                  ', ', num2str(nSample), '/', num2str(totalSample), '.']);
        end

        % initialization --------------------------------------------------
        X = cell(T, 1);
        for tTeam = 1 : T
            X(tTeam) = {zeros(Mt(tTeam), K)};
        end
        Y = X;

        % Iteration -------------------------------------------------------
        for iIter = 1 : totalIter
            if (progressBarType == 2) && (mod(iIter, 1000) == 0) %totalIter / 20) == 0)
                disp(['Progress:', num2str(nStepsize), '/', num2str(length(stepsize)), ...
                  ', ', num2str(nSample), '/', num2str(totalSample), ', ', ...
                  num2str(iIter), '/', num2str(totalIter),'.']);
            end

            % Strategy and Estimates of Each Player
            vecXk = cell(K, 1);
            for kPlayer = 1 : K
                singleXk = zeros(M, 1);
                for sTeam = 1 : T
                    sTeamStrat = cell2Mat(X, sTeam);
                    lowerLim   = stratLowerLim(sTeam);
                    upperLim   = stratUpperLim(sTeam);

                    singleXk(lowerLim : upperLim) = sTeamStrat(:, kPlayer);
                end
                vecXk(kPlayer) = {singleXk};
            end
            vecYk = cell(K, 1);
            for kPlayer = 1 : K
                singleYk = zeros(M, 1);
                for sTeam = 1 : T
                    sTeamStrat = cell2Mat(Y, sTeam);
                    lowerLim   = stratLowerLim(sTeam);
                    upperLim   = stratUpperLim(sTeam);

                    singleYk(lowerLim : upperLim) = sTeamStrat(:, kPlayer);
                end
                vecYk(kPlayer) = {singleYk};
            end

            % Compute Gradient: 
            % Used for Updating Strategy and Estimates on Team t's Strategy
            gradient = cell(T, 1);
            gradforY = cell(T, 1);
            for tTeam = 1 : T
                tTeamGrad    = zeros(Mt(tTeam), K);
                tTeamGradY   = tTeamGrad;
                teamLowerLim = playerLowerLim(tTeam);
                teamUpperLim = playerUpperLim(tTeam);
                for kPlayer = 1 : K
                    if (kPlayer >= teamLowerLim) && (kPlayer <= teamUpperLim)
                        tTeamGrad(:, kPlayer) = localGradient(tTeam, kPlayer, cell2Mat(vecXk, kPlayer));
                        tTeamGradY(:, kPlayer) = localGradient(tTeam, kPlayer, cell2Mat(vecYk, kPlayer));
                    else
                        ellPlayer = teamLowerLim + mod(kPlayer, Kt(tTeam));
                        tTeamGrad(:, kPlayer) = localGradient(tTeam, ellPlayer, cell2Mat(vecXk, kPlayer));
                    end
                end
                tTeamGrad       = tTeamGrad + (2*sigma * rand(Mt(tTeam),K) - sigma);
                tTeamGradY      = tTeamGradY + (tTeamGradY~=0) .* (2*sigma * rand(Mt(tTeam),K) - sigma);
                gradient(tTeam) = {tTeamGrad};
                gradforY(tTeam) = {tTeamGradY};
            end
            
            % Recursion: Update State
            for tTeam = 1 : T
                X(tTeam) = {(cell2Mat(X, tTeam) - mu * cell2Mat(gradient, tTeam)) * cell2Mat(B, tTeam)};
                Y(tTeam) = {(cell2Mat(Y, tTeam) - mu * cell2Mat(gradforY, tTeam)) * cell2Mat(B, tTeam)};
            end

            % Recording the Vector of State to Nash Equilibrium
            vecX = zeros(K * M, 1);
            vecY = vecX;
            for tTeam = 1 : T
                teamLowerLim = K * (stratLowerLim(tTeam) - 1) + 1;
                teamUpperLim = K * stratUpperLim(tTeam);
                tTeamStrat   = cell2Mat(X, tTeam);
                tTeamStratY  = cell2Mat(Y, tTeam);

                vecX(teamLowerLim : teamUpperLim) = tTeamStrat(:);
                vecY(teamLowerLim : teamUpperLim) = tTeamStratY(:);
            end
            vecX2NE(:, iIter, nSample) = XStar - vecX;
            vecY2NE(:, iIter, nSample) = XStar - vecY;
        end
    end
    
    % Performance ---------------------------------------------------------
    % taking average of each sample
    avgSquaredDistX2NE = mean(vecnorm(vecX2NE) .^ 2, 3);
    avgSquaredDistY2NE = mean(vecnorm(vecY2NE) .^ 2, 3);
    hold on
    plot(1 : totalIter, avgSquaredDistX2NE, 'DisplayName', ['$\mu = ', num2str(mu), '$, w/ local knowledge.']);
    plot(1 : totalIter, avgSquaredDistY2NE, 'DisplayName', ['$\mu = ', num2str(mu), '$, w/o local knowledge.']);
    hold off
    
    % store performance of stepsize
    distX2NE(nStepsize) = max(avgSquaredDistX2NE(end - totalIter / 10 : end));
end

%% Plotting Results
grid on;
set(gca, 'YScale', 'log');
title('Convergence to Nash Equilibrium');
xlabel('Iteration');
ylabel('$||x-x^\star||^2$', 'Interpreter', 'latex');
legend('Interpreter', 'latex');
fontname('Times New Roman');

% Exponent to Convergence Radius
beta  = pinv([ones(length(stepsize), 1), log(stepsize')]) * log(distX2NE');
slope = beta(2);
disp(['Convergence radius is of size O(\mu^(', num2str(slope), ')).']);
figure;
plot(stepsize,distX2NE);
grid on;
set(gca,'XScale','log','YScale','log');
title('Distance to Nash Equilibrium')
xlabel('$\mu$','Interpreter','latex');
ylabel('$\limsup||x-x^\star||^2$', 'Interpreter', 'latex');
set(gcf,'color',[1,1,1]);
fontname('Times New Roman');

%% Functions
function mat = cellToMatrixConversion(cellOfMatrices, index)
% CELLTOMATRIXCONVERSION Outputs the matrix in a cell.
% 
% Input:
%   cellOfMatrices      A cell array containing many matrices.
%   index               The index to the desired cell.
% 
% Output:
%   mat                 The matrix in the desired cell.

if length(index) == 2
    mat = cell2mat(cellOfMatrices(index(1), index(2)));
else
    mat = cell2mat(cellOfMatrices(index));
end
end

function localGradient = computeLocalGradient(tTeam, kPlayer, x, Mt, Ak, bk, Ck)
% COMPUTELOCALGRADIETN Computes the specified local gradient vector.
%
% Input:
%   tTeam               The team the gradient is based on.
%   kPlayer             The player whose information is used.
%   x                   The strategy and estimates of said player.
%   Mt                  The strategy size of all teams.
%   Ak                  (Loss function parameters.)
%   bk                  (Loss function parameters.)
%   Ck                  (Loss function parameters.)
%
% Output:
%   localGradient       The desired local gradient of team 'tTeam',
%                       using the information from 'kPlayer'.

cell2Mat = @(cellOfMatrices, index) cellToMatrixConversion(cellOfMatrices, index);

strategyLowerLim = sum(Mt(1 : tTeam - 1)) + 1;
strategyUpperLim = sum(Mt(1 : tTeam));

localGradient = cell2Mat(Ak, kPlayer) * x(strategyLowerLim : strategyUpperLim) ...
                + cell2Mat(bk, kPlayer) + cell2Mat(Ck, kPlayer) * x;
end

function globalGradient = computeGlobalGradient(p, x, Kt, Mt, Ak, bk, Ck)
cell2Mat       = @(cellOfMatrices, index) cellToMatrixConversion(cellOfMatrices, index);
localGradient  = @(tTeam, kPlayer, x) computeLocalGradient(tTeam, kPlayer, x, Mt, Ak, bk, Ck);

globalGradient = zeros(sum(Mt),1);
T              = length(Mt);
for tTeam = 1 : T
    pt = cell2Mat(p, tTeam);

    teamGradient   = 0;
    playerLowerLim = sum(Kt(1 : tTeam - 1));
    for k = 1 : length(pt)
        teamGradient = teamGradient + pt(k) * localGradient(tTeam, playerLowerLim + k, x);
    end

    stratLowerLim = sum(Mt(1 : tTeam - 1)) + 1;
    stratUpperLim = sum(Mt(1 : tTeam));

    globalGradient(stratLowerLim : stratUpperLim) = teamGradient;
end
end

function X = solveDTLyapunov(B,Y)
% Solve the discrete-time Lyapunov equation of the form
%       X - B * X * B' = Y.
% Input:
%       B       A square matrix.
%       Y       A sqaure matrix of same size as 'B'.
% Output:
%       X       The solution to the Lyapunov equation.

    N = size(B,1) ^ 2;
    X = inv(eye(N) - kron(B', B')) * Y(:);
    X = reshape(X,size(B));
end

function W = solveCTLyapunov(D,Y)
% Solve the continuous-time Lyapunov equation of the form
%       WD + D'W = Y.
% Input:
%       D       A square matrix.
%       Y       A sqaure matrix of same size as 'D'.
% Output:
%       W       The solution to the Lyapunov equation.

    N = size(D,1);
    W = inv(kron(D', eye(N)) + kron(eye(N), D')) * Y(:);
    W = reshape(W,size(D));
end