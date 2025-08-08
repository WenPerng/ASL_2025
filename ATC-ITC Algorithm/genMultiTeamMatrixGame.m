% Updated: 0803
% This code generates a matrix game automatically.
% There are K1 + K2 players.
% The strategies for the K1/2's are of size M1/2.

function genMultiTeamMatrixGame(Kt, Mt, lambda)
T = length(Kt);

%% Initialization
% Function handles
PD = @(k) PositiveDefinite(k, lambda);

% Variable allocation
K  = sum(Kt);
Ak = cell(K, 1);
bk = cell(K, 1);
Ck = cell(K, 1);

%% Generation
for t = 1 : T
    playerLowerLim = sum(Kt(1 : t - 1)) + 1;
    playerUpperLim = sum(Kt(1 : t));
    for k = playerLowerLim : playerUpperLim
        Ak(k)  = {PD(Mt(t))};
        bk(k)  = {0.1 * rand(Mt(t), 1) + 4};%{5 * ones(Mt(t),1)};
        C_temp = 0.1 * rand(Mt(t), sum(Mt)) - 1;%-ones(Mt(t), sum(Mt));

        stratLowerLim = sum(Mt(1 : t - 1)) + 1;
        stratUpperLim = sum(Mt(1 : t));
        C_temp(:, stratLowerLim : stratUpperLim) = 0;
        Ck(k) = {C_temp}; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end

%% Save Data
save('Matrix_game_data_multiTeam.mat', "Ak", "bk", "Ck");
end

%% Functions
function X = PositiveDefinite(k, lambda)
    D = diag(0.1 * rand(k, 1) + lambda);
    U = rand(k, k);
    U = U - U';
    U = expm(U);
    X = U' * D * U;
end