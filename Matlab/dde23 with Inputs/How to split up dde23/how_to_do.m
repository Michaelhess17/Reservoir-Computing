clear;

[sox_b, soy_b] = broken_MG_dde23(2);
[sox, soy] = MG_dde23(2);

subplot(1,2,1);
plot(sox_b, soy_b);
title("dde23 Broken Attempt");

subplot(1,2,2);
plot(sox,soy);
title("dde23 Normal");
