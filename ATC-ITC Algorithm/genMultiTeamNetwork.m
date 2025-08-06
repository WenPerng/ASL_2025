function genMultiTeamNetwork(K)
%% Initialization
T = length(K);
A = cell(T, 1);
C = cell(T, T);

%% Network Structure
switch T
    case 3
        % combination matrices (within-team)
        A1 = [1/3,1/2,1/2;
              1/3,1/2,  0;
              1/3,  0,1/2];
        A2 = [1/2,1/3,  0;
              1/2,1/3,1/2;
                0,1/3,1/2];
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
                  0,   0,   0;
                  0,   0,   0;];
        C31 = C21;
        % team 2
        C22 = [9/20, 1/3,   0;
               9/20, 1/3, 1/2;
                  0, 1/3, 1/2];
        C12 = C21;
        C32 = C21;
        % team 3
        C33 = C11;
        C13 = C21;
        C23 = C21;

        C(1,1) = {C11};
        C(1,2) = {C12};
        C(1,3) = {C13};
        C(2,1) = {C21};
        C(2,2) = {C22};
        C(2,3) = {C23};
        C(3,1) = {C31};
        C(3,2) = {C32};
        C(3,3) = {C33};

    case 2
        % combination matrices (within-team)
        A1 = [1/3, 1/2, 1/2;
              1/3, 1/2,   0;
              1/3,   0, 1/2];
        A2 = [1/2, 1/3,   0;
              1/2, 1/3, 1/2;
                0, 1/3, 1/2];
        A  = {A1, A2};
        
        % inference matrices (cross-team)
        C11 = [3/10, 1/2, 1/2;
               3/10, 1/2,   0;
               3/10,   0, 1/2];
        C12 = [1/10,   0,   0;
                  0,   0,   0;
                  0,   0,   0;];
        C21 = C12;
        C22 = [9/20, 1/3,   0;
               9/20, 1/3, 1/2;
                  0, 1/3, 1/2];
        C = {C11, C12; C21, C22};
    otherwise
        disp(['Size of network not implemented yet! ', ...
              'Hence, a crude non-random network consisting of ', ...
              num2str(T), ' teams is given.']);
        Atemp = [1/3,1/2,1/2;
                 1/3,1/2,  0;
                 1/3,  0,1/2];
        Ctemp = [3/10, 1/2, 1/2;
                 3/10, 1/2,   0;
                 3/10,   0, 1/2];
        Dtemp = [1/10,   0,   0;
                    0,   0,   0;
                    0,   0,   0;];

        for tTeam = 1 : T
            A(tTeam) = {Atemp};
            for sTeam = 1 : T
                if sTeam == tTeam
                    C(sTeam,tTeam) = {Ctemp};
                else
                    C(sTeam,tTeam) = {Dtemp};
                end
            end
        end
end

%% Save Data
save('Network_structure_multiTeam.mat',"A","C");

end