function [sox,soy] = dde23Hayes(tau)
    % Solves Mackey-Glass with DDE23; changing tau and returns y outputs (w/ corresponding t outputs).
    %
    % Where Z is the the delayed x, x is the current x, t is the
    % current time and XL is the output of the system tau time ago.

    % Declare variables -> Pulled from figure 4 "Quantization Improves
        % Stabilization of dynamical systems with delayed feedback"
    a = 1;
    C = 3;  % Should get some unstability around tau = 0.7ish with these parameters

    % Using Hayes equation
    dx = @(t,x,XL) a .* x - C .* XL;

    duration = 1000;                    % default set to 1000 seconds
    iter = floor(duration ./ tau);      % how many iterations of tau fit within duration
    t = linspace(0,duration,iter);      % Cut up a range from 0 to duration by the amount of iteration
    
    hist = 0.01;                        % Initial Condition
    sol = dde23(dx, tau, hist, t);
    soy = sol.y;
    sox = sol.x;

end