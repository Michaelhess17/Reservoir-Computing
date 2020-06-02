
%% TESTS

    %% TEST Sampling Different Points

    % count = 1;
    % input = [0.9,0.1,0.001,0.0001,0.00001];         % At this moment, seems like 0.00001 is the lowest magnitude allowed
    % for n = input
    %     subplot(size(input,2), 1, count);
    %     MGEulerComp(2, 2, 9.6, 1, 300, n, 1.3)
    %     count = count + 1
    % end
    
    %% TEST MGEulercomp

    % d,e = MGEulerComp(2, 2, 9.6, 1, 300, 0.001, 1.3)

    %% TEST dde23comp

    % d = dde23Comp(Inputsequence,1,1,2,1,7,0.2);

    %% TEST Comparison Functions

%     d = ddeVEuler(0.02,2,false);

%     d = ddeVEuler(0.01,2,true);

    % Value not Working ->  eulert start point changed to 0
    % d = ddeVEuler(0.400750375187594,2,false);     

    % subplot(1,2,1);
    % [d,t] = dde23Comp(2,2,9.65,1,300, 1.3);
    % subplot(1,2,2);
    % d = MGEulerComp(2, 2, 9.6, 1, 300, .0000500, 1.3);


%% COMPARISON LOOPS
    %% Calculate and graph 0.000001 -> 0.1

    % testRange = linspace(0.000001,0.1,200);
    % counter = 1;
    % 
    % for n = testRange
    %     spError(counter) = ddeVEuler(n,2,false);
    %     counter = counter +1;
    % end
    % 
    % plot(testRange(:), spError);
    % xlabel("Integration Step");
    % ylabel("squared error");
    % title("error for different euler integration steps")

    %% 2000 points between 0.00001 -> 1
%       
%     NumofSteps = 8000;
%     min = 0.00001;
%     max = 1;
%     testRange = linspace(min,max, NumofSteps);
%     counter = 1;
%     tau = 2;
% 
%     for n = testRange
%         [spError(counter), eta, p, gamma, iterations, initial, spTime(counter)] = ddeVEuler(n,tau,false);
%         counter = counter +1;
%     end
%     
%     subplot(2,1,1);
%     plot(testRange(:), spError);
%     xlabel("Integration Step");
%     ylabel("Squared error");
%     title("squared error over "+ num2str(NumofSteps) + " euler integration steps from " + num2str(min) + " to " + num2str(max))
%     a = gca;
%     a.Position(3) = 0.60;
%     annotation( 'textbox', ...
%         [ 0.75, 0.85, 0.1, 0.1], 'String', ["dx = -gamma * x(t) + eta *(x(t-tau)/ (1 + x(t-tau) ** p))" , "eta=" + num2str(eta)  "p =" + num2str(p) "gamma=" + num2str(gamma)...
%          "Iterations of tau = " + num2str(iterations)  "Initialvalue = " + num2str(initial)...
%          "tau =" + num2str(tau)]);
%      
%     subplot(2,1,2);
%     plot(testRange(:), spTime);
%     b = gca;
%     b.Position(3) = 0.6;
%     xlabel("Integration Step");
%     ylabel("time(s) needed for Euler computation");

   

    %% Quick and Dirty Version-testing
    
%     NumofSteps = 2;
%     min = 0.01;
%     max = 1;
%     testRange = linspace(min,max, NumofSteps);
%     counter = 1;
%     tau = 2;
% 
%     for n = testRange
%         [spError(counter), eta, p, gamma, iterations, initial, spTime(counter)] = ddeVEuler(n,tau,false);
%         counter = counter +1;
%     end
%     
%     subplot(2,1,1);
%     plot(testRange(:), spError);
%     xlabel("Integration Step");
%     ylabel("Squared error");
%     title("squared error over "+ num2str(NumofSteps) + " euler integration steps from " + num2str(min) + " to " + num2str(max))
%     a = gca;
%     a.Position(3) = 0.60;
%     annotation( 'textbox', ...
%         [ 0.75, 0.85, 0.1, 0.1], 'String', ["dx = -gamma * x(t) + eta *(x(t-tau)/ (1 + x(t-tau) ** p))" , "eta=" + num2str(eta)  "p =" + num2str(p) "gamma=" + num2str(gamma)...
%          "Iterations of tau = " + num2str(iterations)  "Initialvalue = " + num2str(initial)...
%          "tau =" + num2str(tau)]);
%      
%     subplot(2,1,2);
%     plot(testRange(:), spTime);
%     b = gca;
%     b.Position(3) = 0.6;
%     xlabel("Integration Step");
%     ylabel("time(s) needed for Euler computation");

%% Compare the graphs of different peaks and mins around the end of p = 7
% Near the lower limits of eta = 2, p = 7, gamma = 1, iterations of tau =
    % 200, initialvalue = 1.3, tau = 2, we see a bunch of oscillationsin
    % the squared error. We want to plot some solutions of the max and dips of int steps there
    
% Find and pluck some of the peaks/min
    tau = 2;
    theta = 0.1053;     %peak plucked from "Eulervdde8000p7.fig"
    figure(1);
    [spError, eta, p, gamma, iterations, initial, spTime] = ddeVEuler(theta,tau,true);
    
    theta = 0.104;     %dip plucked from "Eulervdde8000p7.fig"
    figure(2);
    [spError, eta, p, gamma, iterations, initial, spTime] = ddeVEuler(theta,tau,true);
    
    theta = 1.00e-05;   % Most accurate choice of integration step for comparision.
    figure(3);
    [spError, eta, p, gamma, iterations, initial, spTime] = ddeVEuler(theta,tau,true);
%% FUNCTIONS

function [eulert, M_x] = MGEulerComp( tau, eta, p, gamma, iterations, theta, initial)
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
    

    N = floor(tau ./ theta);            % Determine how many appoximate nodes we can fit given some theta
    M_x = ones(1,N) .* initial;         % Set first row(initial conditions) to some value

    %Calculates for iterations after the initial conditions
    for i = 2:iterations + 1
        for j = 0       %Used to evaluate MG for the first value in a new column
            vn_0 = M_x(i-1,end) + (- gamma .* M_x(i-1,end) + (eta .* M_x(i-1,end)) ...
                ./ (1 + (M_x(i-1,end)).^ p)) .* theta;
            M_x(i,1) = vn_0;
        end

        for j = 2:N   % Evaluates the nodes other than first value
            vn = M_x(i,j - 1) + (- gamma .*  M_x(i, j-1) + ...
                (eta *  M_x(i - 1 , j - 1)) ...
                ./ (1 + (M_x(i-1, j-1)).^ p )) .* theta;
            M_x(i, j) = vn;
        end
    end
    M_x = M_x(2:end,:);      % Delete the first row of initial conditions
    M_x = reshape(M_x.',1,[]);      % Reshape solution array into vector
    
    % Get the t coordinates that match with each value of M_x
    total = tau .* (iterations);
    eulert = [0: total ./ (N .* (iterations)): total];          % Change the start to 0 from theta
    M_x = [initial M_x];                                        % Adds in Initial condition into history
    
%     plot(eulert, M_x); Sanity Test Plot
   
end

function [tspace,yspace] = dde23Comp(tau, eta, p, gamma, iterations,initial)
    % Solves the MG equation using dde23
    %
    % args:
    % tau = time delay
    % eta = parameter in front of delayed term 
    % p = exponent
    % gamma = multiplies the solution at x(t)
    % iterations = how many iterations of tau?
    % initial = initial condition
    %
    % returns:
    % tspace = array containing the time coordinates corresponding to solutions
    % yspace = array containing the solutions at ^
    %
    % Where XL is the the delayed x, x is the current x and t is the
    % current time.
    
    
    % Define the equation
    dx = @(t,x,XL)  eta .* ( XL ./ ( 1 + XL .^ p) ) - (gamma .* x);

    % Define pieces for dde23
    duration = iterations .* tau;           % how many cycles of tau do you want?
    t = linspace(0,duration,duration);  
    
    hist = initial;                             % x(t < 0) = 0
    sol = dde23(dx, tau, hist, t);
    
    % Set outputs
    tspace = linspace(0,duration,1000);     %Cut up the space we evaluate at into 1000 chunks
    yspace = deval(sol,tspace);
    
end


function [diff, eta, p, gamma, iterations, initial, tElapsed] = ddeVEuler(theta, tau, plotOn)
    % ddeVEuler compares the accuracy of Euler method of integration
    % to method of steps (dde23). Varies the time step of Euler.

    % args:
    % theta = integration steps to calculate euler
    % tau = time delay
    % plotOn = true/false -> Option to plot the results of some integration
        % step
    %
    % returns:
    % diff = array of mean squared error between the two methods.
    % eta, p, gamma, iterations, initial = parameters for mg used
    % tElapsed = array time needed to compute mg for some integration step
    
    
    % Define variables
    eta = 2;            % mult. in front of delayed term
    p = 1;            % exponent
    gamma = 1;          % in front of x(t) term
    iterations = 200;   % how many iterations of tau do you want? 
    initial = 1.3;      % initial value
    
    % Call both functions 
    tic;        % Measure the time it takes for euler to calculate the solution
    [eulert, eulerSol] = MGEulerComp(tau, eta, p, gamma, iterations, theta, initial);
    tElapsed = toc;
    [ddeSolt,ddeSoly] = dde23Comp( tau, eta, p, gamma, iterations, initial);
    
    % Find the values at certain x values (matching with dde solutions) of Euler solution to compare 
    EulerPts = interp1(eulert, eulerSol, ddeSolt);
    
    % Compare results and calculate total sq. error
    error = EulerPts - ddeSoly;
    SqError = error .^ 2;
    size(error);
    diff = mean(mean(SqError,1),2);

    % Plot it all out for one value given "true"
    if plotOn == true
        figure1 = figure;

        subplot(2,1,1);             % plot dde23's solutions
        plot(ddeSolt,ddeSoly,"--");
        hold on;
                                    % plot euler solutions
        plot(eulert ,eulerSol);
        title("DDE23 vs. Euler")
        hold off;
        legend("DDE23", "Euler");
        ylabel("x(t)");
        xlabel("t");
        a = gca;
        a.Position(3) = 0.60;
        annotation('textbox',[ 0.75, 0.85, 0.1, 0.1], 'String', ["eta=" + num2str(eta)  "p =" + num2str(p)+ ",gamma=" + num2str(gamma)...
             "Iterations of tau = " + num2str(iterations)  "Initialvalue = " + num2str(initial)...
             "Integration step (theta) =" + num2str(theta)  "tau=" + num2str(tau)]);

        subplot(2,1,2)
        plot(ddeSolt, SqError);
        title(["Mean Squared Error = ",num2str(diff)]);
        xlabel("t")
        ylabel("squared error")


%         saveas(figure1,'EulervDDE23.png')
    else
        return
    end

end


