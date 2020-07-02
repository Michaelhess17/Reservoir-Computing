function y_k = NARM_Generator(length,u)

%     Generate NARMA10 Sequence
%
%     args:
%       Length: Length of NARMA10 Series
%       u: Input Data following requirements for stability outlined in https://arxiv.org/pdf/1906.04608.pdf
%     returns: 
%       N10: NARMA10 Series with k = length entries

       y_k = ones(1,10) .* 0.1;     %Generate first 10 entries in series
       
       % Calculate the rest
       sumd = 0;
       for k = 10:length - 1        % Minus one because we have to account for the fact that k = 10 meaning 10 -> end is 1 greater than 10 + 1 -> end (we're generating from 10+1)
           for i = 0:9
            sumd = sumd + y_k(k - i);
           end
           
           t = (0.3 .* y_k(1,k)) + ( (0.05 .* y_k(1,k)) .* sumd ) + ( 1.5 .* u(k) .* u(k - 9) ) + 0.1;       % calculates the next value in the time series
           
           y_k = [y_k t];
           sumd = 0;
          
       end
       
       
end