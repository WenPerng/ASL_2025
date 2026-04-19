function genMultiTeamNetwork_paper()
%% Initialization
K = [3, 2, 3];
T = length(K);
C = cell(T, T);

%% Network Structure
% combination matrices (within-team)
A1 = [1/3,1/2,1/2;
      1/3,1/2,  0;
      1/3,  0,1/2];
A2 = [1/2,3/4;
      1/2,1/4];
A3 = [1/4,1/2,1/3;
      1/4,  0,1/3;
      1/2,1/2,1/3];
A  = {A1,A2,A3};

% inference matrices (cross-team)
% team 1
C11 = [3/10, 1/2, 1/2;
       3/10, 1/2,   0;
       3/10,   0, 1/2];
C21 = [1/10,   0,   0;
          0,   0,   0];
C31 = [   0,   0,   0;
       1/10,   0,   0;
          0,   0,   0];
% team 2
C12 = [  0, 1/3;
         0,   0;
         0,   0];
C22 = [1/3, 1/3;
       2/3, 1/3];
C32 = C12;
% team 3
C13 = C31;
C23 = C21;
C33 = C11;

C(1,1) = {C11};
C(1,2) = {C12};
C(1,3) = {C13};
C(2,1) = {C21};
C(2,2) = {C22};
C(2,3) = {C23};
C(3,1) = {C31};
C(3,2) = {C32};
C(3,3) = {C33};

%% Save Data
save('Network_structure_multiTeam.mat',"A","C");

end