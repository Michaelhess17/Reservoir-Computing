import numpy as np
from matplotlib import pyplot as plt

from helper_files import load_NARMA

u, m, target = load_NARMA(True, N = 1)
u = np.reshape(u,(-1,1))
m = np.reshape(m,(1,-1))

j_t = u@m
j_t = j_t.flatten()

#Plot j(t)
plt.plot( np.linspace(0,400*0.2,1600), j_t, label = "J(t)")
plt.xlabel("seconds (theta = 0.2)")
plt.title("J(t), for system with N = 1")
axes = plt.gca()
# axes.set_xlim([0,400*0.2])

plt.legend()
plt.show()
