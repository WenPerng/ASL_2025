% Updated: 0825
% This code generates a matrix game automatically.
% There are K players, from a total of T teams.
% The strategies for the 't'th team is Mt(t).

function genMultiTeamMatrixGame_paper()
Kt = [3, 2, 3];
Mt = [1, 1, 2];

%% Initialization
% Variable allocation
K  = sum(Kt);
Ak = cell(K, 1);
bk = cell(K, 1);
Ck = cell(K, 1);

%% Generation
% Ak
Ak(1) = {26};
Ak(2) = {24};
Ak(3) = {24};
Ak(4) = {32};
Ak(5) = {30};
Ak(6) = {[22,  0;
           0, 36]};
Ak(7) = {[20,  0;
           0, 40]};
Ak(8) = {[20,  0;
           0, 30]};

% Ck
Ck(1) = {[0, 2,  3,  3]};
Ck(2) = {[0, 1, -1, -1]};
Ck(3) = {[0, 0,  1,  1]};
Ck(4) = {[3, 0, 3, 1]};
Ck(5) = {[3, 0, 3, 2]};
Ck(6) = {[ 0, 1, 0, 0;
          -1, 1, 0, 0]};
Ck(7) = {[-1, 2, 0, 0;
          -1, 2, 0, 0]};
Ck(8) = {[-2, 3, 0, 0;
          -1, 2, 0, 0]};

% bk
bk(1) = {5};
bk(2) = {5};
bk(3) = {4};
bk(4) = {-3};
bk(5) = {-4};
bk(6) = {[-4;  1]};
bk(7) = {[-5; -2]};
bk(8) = {[-5; -2]};

%% Save Data
save('Matrix_game_data_multiTeam.mat', "Ak", "bk", "Ck");
end
