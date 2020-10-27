clc;
clear;
Import_script
%% Meat and Potatoes
% take input and matmul it
N = 2;             % Number of simulated nodes
mask2 = mask2(1:N);
u = Inputsequence * mask2.';
u = reshape(u.',1,[]);

[sox, soy] = dde23_Jt_Hayes(0.2,u);

figure(1);
plot(sox,soy);
title("hayes");

[sox_mg, soy_mg] = dde23_Jt_MG(0.2, u);

figure(2);
plot(sox_mg, soy_mg);
title("MG");



