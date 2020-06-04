%% Narma Tester

    %% Import data from text file
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
    Inputsequence = readtable("/Users/michael/Documents/Github/Reservoir-Computing/Matlab/Narma10 and MGEuler Integrator/Input_sequence.txt", opts);        %Change path

    %% Convert to output type
    Inputsequence = table2array(Inputsequence);

    %% Clear temporary variables
    clear opts
    
    %% Run Narma Sequence Generator
    
    NarmaSeq = NARM_Generator(length(Inputsequence),Inputsequence);
    