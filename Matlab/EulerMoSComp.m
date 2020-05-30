
%TESTS 

%d = dde23Comp(Inputsequence,1,1,2,1,7,0.2);
% d = ddeVEuler(Inputsequence,0.05);

% subplot(1,2,1);
% [d,t] = dde23Comp(2,2,9.65,1,300, 1.3);
% subplot(1,2,2);
% d = MGEulerComp(2, 2, 9.6, 1, 300, .0000500, 1.3);
 
% count = 1;
% input = [0.9,0.1,0.001,0.0001,0.00001];         % At this moment, seems like 0.00001 is the lowest magnitude allowed
% for n = input
%     subplot(size(input,2), 1, count);
%     MGEulerComp(2, 2, 9.6, 1, 300, n, 1.3)
%     count = count + 1
% end

% d,e] = MGEulerComp(2, 2, 9.6, 1, 300, 0.001, 1.3)

% END OF TESTS

 d = ddeVEuler(0.400750375187594,2,false);



% % 0.000001 -> 0.1
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

% % 0.1 -> ~5
% testRange = linspace(0.01,1,2000);
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

function [eulert, M_x] = MGEulerComp( tau, eta, p, gamma, iterations, theta, initial)
    % Calculates output of MG using euler 
    
    % output:
    % M_x: Matrix of reservoir history of size (# tau, Nodes)
    
    N = floor(tau ./ theta);            % Determine how many appoximate nodes we can fit given some theta
    M_x = ones(1,N) .* initial;         % Set first row(initial conditions) to some value

    for i = 2:iterations+1
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
    M_x = reshape(M_x.',1,[]);
    
    % Get the x coordinates that match with each value of M_x
    total = tau .* iterations;
    eulert = [theta: total ./ (N .* iterations): total];
    
     plot(eulert, M_x);
   
end


function [tspace,yspace] = dde23Comp(tau, eta, p, gamma, iterations,initial)
    % Calculates (Without masking) state of the reservoir using DDE 23
    %
    % Where Z is the the delayed x, x is the current x and t is the
    % current time.
    
    % Define vars
    
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
    
%     plot(tspace(:),yspace)
%   plot(sol.x,sol.y);

end


function diff = ddeVEuler(theta, tau, plotOn)
    % ddeVEuler compares the accuracy of Euler method of integration
    % to method of steps (dde23). Varies the time step of Euler.

    % args:
    % input = vertical array of input values
    % theta = time between nodes. If running only a single node, synonymous
    %   to tau
    % plotOn = true/false

    % returns:
    % mean squared error between the two methods.
    
    
    % Define variables
    eta = 2;            % mult. in front of delayed term
    p = 9.6;            % exponent
    gamma = 1;          % in front of x(t) term
    iterations = 200;   % how many iterations of tau do you want? 
    initial = 1.3;      % initial value
    
    
    % Call both functions 
    [eulert, eulerSol] = MGEulerComp(tau, eta, p, gamma, iterations, theta, initial);
    [ddeSolt,ddeSoly] = dde23Comp( tau, eta, p, gamma, iterations, initial);
    
    % Find the values at certain x values of Euler solution to compare 
    EulerPts = interp1(eulert, eulerSol, ddeSolt);
    
    % Compare results and calculate total sq. error
    
    error = EulerPts - ddeSoly;
    error = error(2:end);           % Cut off the first value because euler begins recording at first calculation
    error = error(1:end-1);         % Snip off the end to account for other discontinuities
    SqError = error .^ 2;
    size(error);
    diff = mean(mean(SqError,1),2);
    

    % Plot it all out
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

        subplot(2,1,2)
        plot(ddeSolt(2:end), SqError);
        title(["Mean Squared Error = ",num2str(diff)])


        saveas(figure1,'EulervDDE23.png')
    else
        return
    end

end


