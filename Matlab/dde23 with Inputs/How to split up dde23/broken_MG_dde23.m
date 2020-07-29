function [sox,soy] = broken_MG_dde23(tau)
    % Solves Mackey-Glass with DDE23 in pieces
    %
    % Goal: Be able to inject new j(t) into integration

    % Declare Parameters and initial conditions
    eta = 2;
    gamma = 1; 
    n = 9.65;
    hist = 1.1;
    
    % Define Mackey-Glass equation, duration for entire integration and
    % integration options
    dx = @(t,x,XL)  eta .* ( XL ./ ( 1 + XL .^ n) ) - (gamma .* x);
    duration = 1000;
    opts = ddeset("RelTol",1e-5);
    
    % Run first dde solve on the first block of tau
    sol = dde23(dx, tau, hist, [0,tau]);
    
    while sol.x(end) < duration
        opts = ddeset(opts, 'InitialY', [sol.y(:,end)]);
        sol = dde23(dx, tau, sol, [sol.x(end), sol.x + tau], opts);           % This should solve MG for some interval
%         hist = it_sol.y;          % Because we are "continuing the integration" but only changing the j(t)
        
    end

    sox = sol.x;
    soy = sol.y;
    
end
