function sol = dde23MG(tau)
    % Solves Mackey-Glass with DDE23; chaning tau
    %
    % Where Z is the the delayed x, x is the current x, t is the
    % current time and XL is the output of the system tau time ago.

    % Declare variables
    eta = 2;       
    gamma = 1;   
    n = 9.65;
%     tau = 2;

    % Edit the way that you iterate through the inputs, its sloppy
    dx = @(t,x,XL)  eta .* ( XL ./ ( 1 + XL .^ n) ) - (gamma .* x);

    iter = 20;
    duration = iter .* tau;        % how many tau cycles do you want?
    t = linspace(0,duration,iter);

    hist = 0.5;
    sol = dde23(dx, tau, hist, t);

    plot(sol.x,sol.y);

end

