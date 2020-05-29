d = ddeVEuler(Inputsequence,0.0001);

function M_x = MGEulerComp(input,gamma,N, eta, loops,p,theta)
    % Calculates output of MG using euler WITHOUT
    % RESERVOIR/Masking
%                   - Possibility of upgrading this code to handle multiple loops
    % args:
    % column vector of inputs j(x)
    
    % output:
    % M_x: Matrix of reservoir history of size (# tau, Nodes)
    
    
    J = input * ones(1 , N);
    cycles = size(input);   %The number of rows we have
    M_x = zeros(cycles(1), N);  
    J = [zeros(1,N); J];    % Sets initial Value to 0

    for i = 2:cycles+1
        for j = 0       %Used to evaluate MG for the first value in a new column
            vn_0 = M_x(i-1,end) + (- M_x(i-1,end) + eta .* (M_x(i-1,end) + ...
                    gamma .* J(i-1,end)) ./ (1 + (M_x(i-1,end) + gamma ...
                    .* J(i-1,end)).^ p)) .* theta;
            M_x(i,1) = vn_0;
        end

        for j = 2:N   % Evaluates the nodes other than first value
            vn = M_x(i,j - 1) + (- M_x(i, j - 1) + ...
                eta * ( M_x(i - 1 , j - 1) + gamma .* J(i,j-1) ...
                ./ (1 + (M_x(i-1, j - 1) + gamma .* J(i, j - 1)).^ p ) .* theta));
            M_x(i, j) = vn;
        end
    end
    M_x = M_x(2:end,:);      % Delete the first row of initial conditions

    t = 400;
%             plot(M_x);
end


function sol = dde23Comp(input,gamma,N, eta, loops,p,theta)
    % Calculates (Without masking) state of the reservoir using
    % DDE 23
    %
    % Where Z is the the delayed x, x is the current x and t is the
    % current time.
    J = input;      %Possibility of multiplying against the nodes like above to compare node performace?
    J = [J ; 0];
    inlen = size(input,1);

    tau = theta .* N;
    hist = 0;

    % Edit the way that you iterate through the inputs, its sloppy
    dx = @(t,x,XL) ( eta .* ( XL + gamma .* J(round(t ./ tau)+1) ) ...
        ./ ( 1 + (XL + gamma .* J(round(t ./ tau)+1)) ) .^ p ) - x;

    duration = inlen .* tau;
    t = linspace(0,duration,inlen);
    hist = 0;

    sol = dde23(dx, tau, hist, t);

%             plot(sol.x,sol.y);

end


function diff = ddeVEuler(input, theta)
    % ddeVEuler compares the accuracy of Euler method of integration
    % to method of steps (dde23). Varies the time step of Euler.

    % args:
    % input = vertical array of input values
    % theta = time between nodes. If running only a single node, synonymous
    %   to tau

    % returns:
    % squared error between the two methods.
    
    
    % Define variables
    gamma = 0.5;    % input weight
    N = 1;          % single node
    eta = 1;        % mult. in front of delayed term
    loops = 1;      % number of delay loops in reservoir
    p = 7;          % exponent
    
    % Call both functions 
    eulerSol = MGEulerComp(input,gamma,N, eta, loops,p,theta);
    ddeSol = dde23Comp(input,gamma,N, eta, loops,p,theta);
    
    % Compare results and calculate total sq. error
    error = eulerSol - ddeSol.y;
    SqError = error .^ 2;
    size(error);
    diff = mean(mean(SqError,1),2);

    % Plot it all out
    figure1 = figure

    subplot(2,2,1);
    plot(ddeSol.x,ddeSol.y);
    title("DDE23")

    subplot(2,2,2);
    plot(linspace(0,size(input,1).* theta, size(input,1)) ,eulerSol);
    title("Euler")

    subplot(2,2,3)
    plot(linspace(0,size(input,1).* theta, size(input,1)), SqError);
    title("squared error")

    saveas(figure1,'EulervDDE23.png')

   fprintf('Squared Total Error = %f1.2',diff);

end


