function [sox,soy] = dde23Hayes(tau, input)
    % Solves Mackey-Glass with DDE23; changing tau and returns y outputs (w/ corresponding t outputs).
    %
    % Args:
    % x = current x
    % t is current time
    % XL = output of the system tau time ago
    % input = column vector of input sequence (May add node calculations also?)
    %
    % Output:
    % sox: solution domain
    % soy: solution values

    % Declare variables -> Pulled from figure 4 "Quantization Improves
        % Stabilization of dynamical systems with delayed feedback"
    k1 = 1.15;
    k2 = 1;  % Should get some unstability around tau = 0.7ish with these parameters
    cuts = 10;          % How many 'cuts' over the iteration time domain
    gamma = 0.05;
    Jt = input(1);
    
    % Using Hayes equation
    dx = @(t,x,XL) k1.*x - k2.*( XL + gamma .* Jt);
    
    cycles = size(input,2);          % Grab the # of columns - how many iterations of tau we are doing
    duration = tau * cycles;            % set the duration to however many cycles of tau
%     t = linspace(0,duration,cycles);      % Cut up a range from 0 to duration by the number of cycles
    
    hist = 0.00;            % initial-initial point
    t = linspace(0,tau,cuts);           % first cut 
    
    % Solve for the first theta block to start sol
    sol = dde23(dx, tau, hist, t);
    
    % Define the number of cycles left
    num_cycles = linspace(1,cycles,cycles);
    
    t_domain = linspace(0, duration, cycles);
    
    for i = num_cycles(2:end)
        time_block = t_domain(i);            % Set the time block
        t = linspace(sol.x(end), time_block,cuts);           % Create a slice over which to solve dde23
        
        hist = sol.y(end:-1:cuts);          % Initial Condition set as the last value computed by the last iteration
        Jt = input(i);           % Redefine input 
        
        % Iteratively solve each theta block
        it_sol = dde23(dx, tau, hist, t);
        
        % Grab the first y values of each branch and flip them so they're
        % horizontal
        
        one_column = it_sol.y(:,2);
        it_sol.y = one_column.';
        
        % Combine history solution and current solution 
        sol.x = [sol.x, it_sol.x];
        sol.y = [sol.y, it_sol.y];
        it_sol = [];
    end
    
    soy = sol.y;
    sox = sol.x;

end
