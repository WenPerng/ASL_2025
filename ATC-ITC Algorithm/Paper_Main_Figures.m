%% ATC-ITC over Games between a Network of Mutiple Teams
% This code does not require any further toolbox to run.
%
% This code simulates the performance of the ATC-ITC (adapt-then-combine,
% inference-then-combin) algorithm.
% This is for the paper version
clear; close all; clc;

%% Setting Up the Simulation: Variables, Parameters, and Function Handles
cell2Mat = @(cellOfMatrices, index) cellToMatrixConversion(cellOfMatrices, index);

progressBarType = 2; % 1, 2

color = orderedcolors("gem");

% Descent Parameters ------------------------------------------------------
stepsize    = 0.001 * 2 .^ (+1 : -1 : -1);          % all step-sizes to be tested
totalIter   = 1 * 10 ^ 3;                           % number of iterations
totalSample = 100;                                  % number of samples

% Noise
noise_range = 1.00;     % range of the zero-mean uniformly-dirstributed gradient noise

% Game Parameters ---------------------------------------------------------
Kt = [3, 2, 3];         % number of players in each team
K  = sum(Kt);           % total number of players
Mt = [1, 1, 2];         % strategy size of each team
M  = sum(Mt);           % total dimension of strategies
T  = length(Kt);        % total number of teams

% Block Sizes
playerLowerLim = @(tTeam) sum(Kt(1 : tTeam - 1)) + 1;
playerUpperLim = @(tTeam) sum(Kt(1 : tTeam));
stratLowerLim  = @(tTeam) sum(Mt(1 : tTeam - 1)) + 1;
stratUpperLim  = @(tTeam) sum(Mt(1 : tTeam));

% Network Structure -------------------------------------------------------
genMultiTeamNetwork_paper();
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
genMultiTeamMatrixGame_paper();
load("Matrix_game_data_multiTeam.mat"); 

% Gradient Function Handles
localGradient  = @(tTeam, kPlayer, x) computeLocalGradient(tTeam, kPlayer, x, Mt, Ak, bk, Ck, noise_range);
globalGradient = @(x) computeGlobalGradient(p, x, Kt, Mt, Ak, bk, Ck);

% Satistics ---------------------------------------------------------------
vecX2NE  = zeros(K * M, totalIter, totalSample);
distX2NE = zeros(1, length(stepsize));
dis1X2NE = zeros(1, length(stepsize));
dis4X2NE = zeros(1, length(stepsize));

avgSquaredDistX2NE = zeros(length(stepsize), totalIter);
avg1NormedDistX2NE = zeros(length(stepsize), totalIter);
avg4NormedDistX2NE = zeros(length(stepsize), totalIter);

centroid    = zeros(T * M, totalIter, totalSample);
avgCentroid = zeros(T * M, totalIter);
stdCentroid = zeros(T * M, totalIter);

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
% Figure Initialization ===================================================
figure;
yline(xStar(1), 'g--', 'LineWidth', 2, 'DisplayName', 'Nash equilibrium $x^{1,\star}$');

% Initial conditions ======================================================
X0 = cell(T, 1);
for tTeam = 1 : T
    X0(tTeam) = {0.5 * rand(Mt(tTeam), K) - 0.25};
end

% Recursion ===============================================================
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
            X(tTeam) = X0(tTeam);
        end

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
                    sTeamStrat    = cell2Mat(X, sTeam);
                    lowerLim = stratLowerLim(sTeam);
                    upperLim = stratUpperLim(sTeam);

                    singleXk(lowerLim : upperLim) = sTeamStrat(:, kPlayer);
                end
                vecXk(kPlayer) = {singleXk};
            end

            % Compute Gradient: 
            % Used for Updating Strategy and Estimates on Team t's Strategy
            gradient = cell(T, 1);
            for tTeam = 1 : T
                tTeamGrad = zeros(Mt(tTeam), K);
                teamLowerLim = playerLowerLim(tTeam);
                teamUpperLim = playerUpperLim(tTeam);
                for kPlayer = 1 : K
                    if (kPlayer >= teamLowerLim) && (kPlayer <= teamUpperLim)
                        tTeamGrad(:, kPlayer) = localGradient(tTeam, kPlayer, cell2Mat(vecXk, kPlayer));
                    else
                        ellPlayer = teamLowerLim + mod(kPlayer, Kt(tTeam));
                        tTeamGrad(:, kPlayer) = localGradient(tTeam, ellPlayer, cell2Mat(vecXk, kPlayer));
                    end
                end
                gradient(tTeam) = {tTeamGrad};
            end
            
            % Recursion: Update State
            for tTeam = 1 : T
                X(tTeam) = {(cell2Mat(X, tTeam) - mu * cell2Mat(gradient, tTeam)) * cell2Mat(B, tTeam)};
            end

            % Recording the Vector of State to Nash Equilibrium
            vecX = zeros(K * M, 1);
            for tTeam = 1 : T
                teamLowerLim = K * (stratLowerLim(tTeam) - 1) + 1;
                teamUpperLim = K * stratUpperLim(tTeam);
                tTeamStrat   = cell2Mat(X, tTeam);

                vecX(teamLowerLim : teamUpperLim) = tTeamStrat(:);
            end
            vecX2NE(:, iIter, nSample) = XStar - vecX;
        end
    end
    disp(['Stepsize done: ', num2str(nStepsize), '/', num2str(length(stepsize))]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Performance =========================================================
    % store performance of stepsize (for rate of convergence)
    avgSquaredDistX2NE(nStepsize, :) = mean(vecnorm(vecX2NE) .^ 2, 3) / K;
    avg1NormedDistX2NE(nStepsize, :) = mean(vecnorm(vecX2NE), 3) / K;       %
    avg4NormedDistX2NE(nStepsize, :) = mean(vecnorm(vecX2NE) .^ 4, 3) / K;  %

    distX2NE(nStepsize) = max(avgSquaredDistX2NE(nStepsize, end - totalIter / 10 : end));
    dis1X2NE(nStepsize) = max(avg1NormedDistX2NE(nStepsize, end - totalIter / 10 : end)); %
    dis4X2NE(nStepsize) = max(avg4NormedDistX2NE(nStepsize, end - totalIter / 10 : end)); %

    % Convergence to consensus --------------------------------------------
    % reordering of strategies with size > 1
    for tTeam = 1 : T
        if Mt(tTeam) > 1
            lowerLim = K * (stratLowerLim(tTeam) - 1) + 1;
            upperLim = K * stratUpperLim(tTeam);

            % a row permutation
            transform = zeros(K * Mt(tTeam));
            for kPlayer = 1 : K
                for mStrat = 1 : Mt(tTeam)
                    indexFrom = Mt(tTeam) * (kPlayer - 1) + mStrat;
                    indexTo   = K * (mStrat - 1) + kPlayer;
                    transform(indexTo, indexFrom) = 1;
                end
            end

            % performing the row permutation
            for nSample = 1 : totalSample
                vecX2NE(lowerLim : upperLim, :, nSample) = transform * vecX2NE(lowerLim : upperLim, :, nSample);
            end
        else
        end
    end

    % consensus of each strategy / estimate in a team
    for mStrategy = 1 : M
        for tTeam = 1 : T
            index    = T * (mStrategy - 1) + tTeam;
            lowerLim = K * (mStrategy - 1) + playerLowerLim(tTeam);
            upperLim = K * (mStrategy - 1) + playerUpperLim(tTeam);
            
            centroid(index, :, :) = ...
                mean(vecX2NE(lowerLim : upperLim, :, :), 1);
        end
    end
    avgCentroid = mean(centroid, 3) + kron(xStar, ones(T, 1));
    stdCentroid = std(centroid, [], 3);

    % plotting figure
    hold on;
    plot(1 : totalIter, avgCentroid(1, :), 'color', color(nStepsize, :), 'LineWidth', 1, ...
        'DisplayName', ['$\mu = ', num2str(mu), '$, ', '$x^1$ centroid of team 1']);
    plot(1 : totalIter, avgCentroid(1, :), '--', 'color', color(nStepsize, :), 'LineWidth', 1, ...
        'DisplayName', ['$\mu = ', num2str(mu), '$, ', '$x^1$ centroids of team 2, 3']);
    for mStrategy = 1 : M
        for tTeam = 1 : T
            index = T * (mStrategy - 1) + tTeam;
            % standard deviation
            x = [1 : totalIter, fliplr(1 : totalIter)];
            inBetween = [avgCentroid(index, :) + 1 * stdCentroid(index, :), fliplr(avgCentroid(index, :) - 1 * stdCentroid(index, :))];
            fill(x, inBetween, color(nStepsize, :), EdgeColor = 'none', FaceAlpha = 0.1, HandleVisibility = 'off');
            % average
            if (mStrategy >= stratLowerLim(tTeam)) && (mStrategy <= stratUpperLim(tTeam))
                plot(1 : totalIter, avgCentroid(index, :), 'color', color(nStepsize, :), 'LineWidth', 2, 'HandleVisibility', 'off');
            else
                plot(1 : totalIter, avgCentroid(index, :), '--', 'color', color(nStepsize, :), 'LineWidth', 2, 'HandleVisibility', 'off');
            end
        end
        yline(xStar(mStrategy), 'g--', 'HandleVisibility', 'off', 'LineWidth', 2);
    end
    hold off;
    
end

%% Plotting Results
% Consensus
set(gcf, 'Position', get(0, 'Screensize')); % full screen
ylabel('$x$', 'Interpreter', 'latex');
xlabel('Iteration, $i$', 'Interpreter', 'latex');
legend('Interpreter', 'latex');
% text(600,xStar(1) + 0.04,'$x^{1}$', 'interpreter', 'latex')
% text(400,xStar(2) + 0.04,'$x^{2}$', 'interpreter', 'latex')
% text(300,xStar(3) + 0.05,'$(x^{3})_{1}$', 'interpreter', 'latex')
% text(500,xStar(4) - 0.05,'$(x^{3})_{2}$', 'interpreter', 'latex')
axis([1, totalIter, -0.22, -0.195]);
grid on;
set(gcf, 'Color', [1, 1, 1]);
fontname('Times New Roman');
fontsize(18, 'points');
exportgraphics(gcf, 'consensus.pdf', 'ContentType', 'vector')

% MSD Curve
figure;
hold on;
for nStepsize = 1 : length(stepsize)
    plot(1 : totalIter, avgSquaredDistX2NE(nStepsize, :), 'Color', color(nStepsize, :), ...
        'DisplayName', ['$\mu = ', num2str(stepsize(nStepsize)), '$'], 'LineWidth', 2);
end
hold off;
grid on;
set(gcf, 'Color', [1, 1, 1]);
set(gca, 'YScale', 'log');
xlabel('Iteration, $i$', 'Interpreter', 'latex');
ylabel('$\mathrm{MSE}_{i}$', 'Interpreter', 'latex');
legend('Interpreter', 'latex');
fontname('Times New Roman');
fontsize(14, 'points');
exportgraphics(gcf,'MSE.pdf','ContentType','vector');

% Exponent to Convergence Radius
beta  = pinv([ones(length(stepsize), 1), log(stepsize')]) * log(distX2NE');
slope = beta(2);
disp(['Convergence radius of error^2 is of size O(\mu^(', num2str(slope), ')).']);

figure;
hold on;
muLog = log(linspace(stepsize(1), stepsize(end), 100));
yLog  = beta(1) + beta(2) * muLog;
plot(exp(muLog), exp(yLog), 'b', 'LineWidth', 2);
scatter(stepsize, distX2NE, 'o', 'filled');
hold off;
text(0.9e-3, 4.5e-5, ['Slope:', num2str(slope)], 'interpreter', 'latex');
grid on;
set(gca,'XScale','log','YScale','log');
xlabel('$\mu$','Interpreter','latex');
ylabel('$\limsup \mathrm{MSD}_{i}$', 'Interpreter', 'latex');
set(gcf,'color',[1,1,1]);
fontname('Times New Roman');
fontsize(14, 'points');
exportgraphics(gcf,'exponent.pdf','ContentType','vector');


beta  = pinv([ones(length(stepsize), 1), log(stepsize')]) * log(dis1X2NE');         %
slope = beta(2);                                                                    %
disp(['Convergence radius of error^1  is of size O(\mu^(', num2str(slope), ')).']); %
beta  = pinv([ones(length(stepsize), 1), log(stepsize')]) * log(dis4X2NE');         %
slope = beta(2);                                                                    %
disp(['Convergence radius of error^4 is of size O(\mu^(', num2str(slope), ')).']);  %

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

function localGradient = computeLocalGradient(tTeam, kPlayer, x, Mt, Ak, bk, Ck, noise_range)
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
%   noise_range         The range to the element-wise noise
%
% Output:
%   localGradient       The desired local gradient of team 'tTeam',
%                       using the information from 'kPlayer'.

cell2Mat = @(cellOfMatrices, index) cellToMatrixConversion(cellOfMatrices, index);

strategyLowerLim = sum(Mt(1 : tTeam - 1)) + 1;
strategyUpperLim = sum(Mt(1 : tTeam));

A = cell2Mat(Ak, kPlayer); A = A + (2 * noise_range * rand(size(A)) - noise_range);
b = cell2Mat(bk, kPlayer); b = b + (noise_range * rand(size(b)) - noise_range / 2);
C = cell2Mat(Ck, kPlayer); C = C + (noise_range * rand(size(C)) - noise_range / 2);

localGradient = A * x(strategyLowerLim : strategyUpperLim) + b + C * x;
end

function globalGradient = computeGlobalGradient(p, x, Kt, Mt, Ak, bk, Ck)
cell2Mat       = @(cellOfMatrices, index) cellToMatrixConversion(cellOfMatrices, index);
localGradient  = @(tTeam, kPlayer, x) computeLocalGradient(tTeam, kPlayer, x, Mt, Ak, bk, Ck, 0);

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

