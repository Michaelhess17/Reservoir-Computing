%% Import data from text file
% Script for importing data from the following text file:
%
%    filename: /Users/michael/Documents/Github/Reservoir-Computing/Matlab/dde23 with Inputs/Input_sequence.txt
%
% Auto-generated by MATLAB on 16-Jul-2020 15:38:36

%% Setup the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 1);

% Specify range and delimiter
opts.DataLines = [1, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = "VarName1";
opts.VariableTypes = "double";

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
Inputsequence = readtable("/Users/michael/Documents/Github/Reservoir-Computing/Matlab/dde23 with Inputs/Input_sequence.txt", opts);

%% Convert to output type
Inputsequence = table2array(Inputsequence);

%% Clear temporary variables
clear opts
%% Solve using MGEulerComp
    
% Define vars -> Using the same variables as RC_test.py from Phillip 
tau = 2;
eta = 1.30;
p = 7;
gamma = 1; 
theta = 0.001;      % Integration step
input = Narma.';
scaling = 1;
iterations = length(Inputsequence);     %Should be the same len as the input sequence

% Solve using Euler Solver
[t, soly] = MGEulerComp(tau, eta, p, gamma, iterations, theta, scaling, input);

plot(t, soly);

%% Masking Function
function maskedInput = Mask(input, varargin)
    % Masks Input with either some pattern else randomly
    %
    % args: 
    % input = input data
    % varagin = optional masking pattern as a column vector

    dim = size(input);

    if nargin == 1
      column = dim(1);
      mask = zeros(1,column);
      for i = 1:column
            mask(1,i) = datasample ([1,-1],1);
          end

      mask
      input
      maskedInput = input * mask;

    else
      mask = varargin{1,1};
      mask = mask.'
      maskedInput = input * mask;
    end
    end

%% MG Solver Function
function [eulert, M_x] = MGEulerComp( tau, eta, p, gamma, iterations, theta, scaling, input)
    % Calculates solution of MG eqn using Euler method
    %
    % args:    
    % tau = time delay
    % eta = parameter in front of delayed term 
    % p = exponent
    % gamma = multiplies the solution at x(t)
    % iterations = how many iterations of tau?
    % theta = size of the integration time step
    % intitial = initial condition
    %
    % returns:
    % M_x: Matrix of reservoir history of size (# tau, Nodes)
    % eulert: corresponding t values for M_x
    %
    

    N = floor(tau ./ theta);            % Determine how many appoximate nodes we can fit given some theta -> In another version we can make it so that N is a parameter
    J = input .* ones(length(input), N);
    M_x = ones(1,N) .* J(1);         % Set first row(initial conditions) to the first value of the input sequence

    %Calculates for iterations after the initial conditions
    for i = 2:iterations   
        for j = 0       %Used to evaluate MG for the first value in a new column
            vn_0 = M_x(i-1,end) + (- gamma .* M_x(i-1,end) + (eta .* M_x(i-1,end)) ...
                ./ (1 + (M_x(i-1,end) + scaling .* J(i-1,end)).^ p)) .* theta;
            M_x(i,1) = vn_0;
        end

        for j = 2:N   % Evaluates the nodes other than first value
            vn = M_x(i,j - 1) + (- gamma .*  M_x(i, j-1) + ...
                (eta *  M_x(i - 1 , j - 1)) ...
                ./ (1 + (M_x(i-1, j-1) + scaling .* J(i,j - 1)).^ p )) .* theta;
            M_x(i, j) = vn;
        end
    end
    M_x = M_x(2:end,:);      % Delete the first row of initial conditions
    M_x = reshape(M_x.',1,[]);      % Reshape solution array into a row vector
    
    % Get the t coordinates that match with each value of M_x
    total = tau .* (iterations) - tau;                               % Delete one row to match above ^ deleting of initial condition row.
    eulert = [theta: theta: total];       % Change the start to theta from 0
    % M_x = [input(1,1) M_x];                                        % Adds in Initial condition into history 
    
%     plot(eulert, M_x); Sanity Test Plot
   
end