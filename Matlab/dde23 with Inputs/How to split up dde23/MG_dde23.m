function [sox,soy] = dde23MG(tau)
    % Solves Mackey-Glass with DDE23; changing tau and returns y outputs (w/ corresponding t outputs).
    %
    % Where Z is the the delayed x, x is the current x, t is the
    % current time and XL is the output of the system tau time ago.

    % Declare variables
    eta = 2;       
    gamma = 1;   
    n = 9.65;

    % Using Mackey-Glass version w/o inputs
    dx = @(t,x,XL)  eta .* ( XL ./ ( 1 + XL .^ n) ) - (gamma .* x);

    duration = 1000;                    % default set to 1000 seconds
    iter = floor(duration ./ tau);      % how many iterations of tau fit within duration
    t = linspace(0,duration,iter);      % Cut up a range from 0 to duration by the amount of iteration
    
    hist = 1.1;                         % Initial Condition
    sol = dde23(dx, tau, hist, t);
    soy = sol.y;
    sox = sol.x;

end

