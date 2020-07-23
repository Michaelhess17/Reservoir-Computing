import numpy as np
import matplotlib.pyplot as plt

from Modified_Delay_Reservoir import mod_Delay_Res, hayes_special_Delay_Res
from helper_files import load_NARMA

"""
Goal of this script is to spit out outputs for calculate given some 
set of parameters for mackey glass and hayes and compare them against
the results from a likely incorrect implementation of dde23 in matlab 
that allows inputs to be injected into the solver. The goal is
to see whether the results of euler in python (which do a relatively good job)
are drastically different from those in matlab. Comparing the matlab results 
to the ones in python will gives us a ballpark of whether or not the thing is working.
"""

def single_node_calculate():

    activation = ["hayes","mg"]
    calc_results = {}

    for activate in activation:
        # Set parameters based on activation used
        if activate == "hayes":
            delay_res = hayes_special_Delay_Res(
                k1 = 1.15,
                N = 400,
                eta = 1,
                gamma = 0.05,
                theta = 0.2,
                beta = 1,
                tau = 400,         # 1 being 1 theta back. here i'm assuming theta = 0.2
                )
        else:
            delay_res = mod_Delay_Res(
                k1 = 1.15,
                N = 400,
                eta = 1,
                gamma = 0.05,
                theta = 0.2,
                beta = 1,
                tau = 400,         # 1 being 1 theta back. here i'm assuming theta = 0.2
                )
        # Load in data
        preload = True
        train_length = 800
        test_length = 800
        N = 1
        u, m, target = load_NARMA(preload, train_length, test_length, N)
        # m = np.eye(u.shape[0])          # Actually, we want m to be an identity matrix!

        # Store results of calculate
        calc_results[activate] = delay_res.calculate(u, m, np.inf, t = 1, act = activate)
    
    # display the results
    calc_hayes = calc_results["hayes"]
    calc_mg = calc_results["mg"]

    plt.figure(1)
    plt.plot(calc_hayes.flatten(), label = "hayes")
    plt.plot(calc_mg.flatten(), label = "mg")
    plt.title("hayes + mg output with one node")
    plt.legend()
    plt.show()

single_node_calculate()