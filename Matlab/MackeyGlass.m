classdef MackeyGlass
    %Defines a class of 
    %   Detailed explanation goes here
    
    properties
        N {mustBeNumeric} % Number of virtual nodes
        eta {mustBeNumeric} % MG Feedback strength -> Not including coupling factor
        gamma {mustBeNumeric} % input scaling
        theta {mustBeNumeric} % Distance between virtual nodes
        loops {mustBeNumeric} % Number of delay loops in reservoir - Usually set to one... unless we want to try more!
        p {mustBeNumeric} %Exponent
    end
    
    methods
        
        
        function obj = MackeyGlass(N,eta,gamma,theta,loops,p)
            %Constructor of an instance of MackeyGlass Class
            %{
                Arguements: N % Number of virtual nodes
                eta MG Feedback strength
                gamma input scaling
                theta Distance between virtual nodes
                loops 
                Number of delay loops in reservoir - Assummed to be one, unless otherwise specified
            %}
            
            obj.N = N;
            obj.eta = eta;
            obj.gamma = gamma;
            obj.theta = theta;
            obj.loops = loops;
            obj.p = p;
        end
        
        
        function maskedInput = mask(obj, input, varargin)
            % Takes and applies specific mask to input
            % Input and mask are column vectors
            % If no mask given, will generates a random mask and outputs that matrix 
            % 
   
            dim = size(input);

            if nargin == 2
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
        
        
        function M_x = MGEuler(obj,input, mask)
            % Calculates the value of each point using euler method
%                   - Possibility of upgrading this code to handle multiple loops
            % M_x: Matrix of reservoir history
            
            cycles = size(input);   %The number of rows we have
            M_x = zeros(cycles(1), obj.N);  
          %Insert portion to scale to voltage range here
            J = obj.mask(input, mask);
            J = [zeros(1,obj.N), J];    % Sets initial Value to 0
            
            for i = 2:cycles+1
                for j = 0       %Used to evaluate MG for the first value in a new column
                    vn_0 = M_x(i-1,end) + (- M_x(i-1,end) + obj.eta .* (M_x(i-1,end) + ...
                            obj.gamma .* J(i-1,end)) ./ (1 + (M_x(i-1,end) + obj.gamma ...
                            .* J(i-1,end)).^ obj.p)) .* obj.theta;
                    M_x(i,1) = vn_0;
                end
                        
                for j = 2:obj.N   % Evaluates the nodes other than first value
                    vn = M_x(i,j - 1) + (- M_x(i, j - 1) + ...
                        self.eta * ( M_x(i - 1 , j - 1) + obj.gamma .* J(i,j-1) ...
                        ./ (1 + (M_x(i-1, j - 1) + obj.gamma .* J(i, j - 1)).^ obj.p ) .* obj.theta)) ;
                    M_x(i, j) = vn;
                end
            end
                  % Delete the first row of initial conditions
        end
        
        function outputArg = MGDDE23;
            
        end
        
        
        function M_x = MGEulerComp(obj,input)
            % Calculates output of MG using euler WITHOUT
            % RESERVOIR/Masking
%                   - Possibility of upgrading this code to handle multiple loops
            % M_x: Matrix of reservoir history
            
            J = input * ones(1 , obj.N);
            cycles = size(input);   %The number of rows we have
            M_x = zeros(cycles(1), obj.N);  
            J = [zeros(1,obj.N); J];    % Sets initial Value to 0
            
            for i = 2:cycles+1
                for j = 0       %Used to evaluate MG for the first value in a new column
                    vn_0 = M_x(i-1,end) + (- M_x(i-1,end) + obj.eta .* (M_x(i-1,end) + ...
                            obj.gamma .* J(i-1,end)) ./ (1 + (M_x(i-1,end) + obj.gamma ...
                            .* J(i-1,end)).^ obj.p)) .* obj.theta;
                    M_x(i,1) = vn_0;
                end
                        
                for j = 2:obj.N   % Evaluates the nodes other than first value
                    vn = M_x(i,j - 1) + (- M_x(i, j - 1) + ...
                        obj.eta * ( M_x(i - 1 , j - 1) + obj.gamma .* J(i,j-1) ...
                        ./ (1 + (M_x(i-1, j - 1) + obj.gamma .* J(i, j - 1)).^ obj.p ) .* obj.theta));
                    M_x(i, j) = vn;
                end
            end
            M_x = M_x(2:end,:);      % Delete the first row of initial conditions
            
            t = 400;
%             plot(M_x);
        end
        
        
        function sol = dde23Comp(obj,input)
            % Calculates (Without masking) state of the reservoir using
            % DDE 23
            %
            % Where Z is the the delayed x, x is the current x and t is the
            % current time.
            J = input;      %Possibility of multiplying against the nodes like above to compare node performace?
            J = [J ; 0];
            inlen = size(input,1);
            
            tau = obj.theta .* obj.N;
            hist = 0;
            
            % Edit the way that you iterate through the inputs, its sloppy
            dx = @(t,x,XL) ( obj.eta .* ( XL + obj.gamma .* J(round(t ./ tau)+1) ) ...
                ./ ( 1 + (XL + obj.gamma .* J(round(t ./ tau)+1)) ) ) - x;

            duration = inlen .* tau;
            t = linspace(0,duration,inlen);
            hist = 0;

            sol = dde23(dx, tau, hist, t);

%             plot(sol.x,sol.y);
            
        end
        
        
        function diff = ddeEulerComp(obj, input)
            
            eulerSol = obj.MGEulerComp(input);
            ddeSol = obj.dde23Comp(input);
            error = ddeSol.y - eulerSol;
            size(error)
%             plot(ddeSol.x, error);
            subplot(1,2,1);

            plot(ddeSol.x,ddeSol.y);
            title("DDE23")
            subplot(1,2,2);
            plot(eulerSol);
            title("Euler")
            
            
            diff = mean(mean(error,1),1)
        end
    
    end
end

