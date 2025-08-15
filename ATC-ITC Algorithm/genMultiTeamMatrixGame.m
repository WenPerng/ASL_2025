% Updated: 0811
% This code generates a matrix game automatically.
% There are K players, from a total of T teams.
% The strategies for the 't'th team is Mt(t).

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
    
    tempATeam = PD(Mt(t));
    tempBTeam = 1 * rand(Mt(t), 1) + 4;
    tempCTeam = 1 * rand(Mt(t), sum(Mt)) - 1;

    stratLowerLim = sum(Mt(1 : t - 1)) + 1;
    stratUpperLim = sum(Mt(1 : t));

    for k = playerLowerLim : playerUpperLim
        Ak(k) = {tempATeam};
        bk(k) = {tempBTeam + 0.2 * rand(Mt(t), 1) - 0.1};

        tempCk = tempCTeam  + 0.2 * rand(Mt(t), sum(Mt)) - 0.1;
        tempCk(:, stratLowerLim : stratUpperLim) = 0;
        Ck(k) = {tempCk};
    end
end

%% Save Data
save('Matrix_game_data_multiTeam.mat', "Ak", "bk", "Ck");
end

%% Functions
function X = PositiveDefinite(k, lambda)
    D = diag(1 * rand(k, 1) + lambda);%diag(lambda * ones(k, 1));
    U = rand(k, k);
    U = U - U';
    U = expm(U);
    X = U' * D * U;
end