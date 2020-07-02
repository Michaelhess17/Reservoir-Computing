 #!/usr/bin/env python3

__original_author__ = "Philip Jacobson"
__email__ = "philip_jacobson@berkeley.edu"

__editors__ = 'Michael Lee and Michael Hess'
__MH_email__ = "mhess21@cmc.edu"
__ML_email__ = "mlee22@cmc.edu"

import numpy as np
import random
import multiprocessing
import csv
import time
import pandas
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
from tqdm import tqdm
import signal
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from Delay_Reservoir import DelayReservoir
from joblib import Parallel, delayed

import pandas as pd
import seaborn as sns
from hyperopt import fmin, tpe, hp
# from hyperopt import hp, tpe, fmin


############################################################################

def NARMA_Generator(length,u):
    """
    Generates NARMA10 Sequence

    Args:
        length: Length of series
        u: Input data

    Returns:
        y_k: NARMA10 series with k = length entries
    """

    #Generate first ten entries in series
    y_k = np.ones(10)*0.1

    #Iteratively calculate based on NARMA10 formula
    for k in range(10,length):
        t = 0.3*y_k[k-1]+0.05*y_k[k-1]*sum(y_k[k-1-i] for i in range(10))+1.5*\
        u[k-1]*u[k-10]+0.1
        y_k = np.append(y_k,t)

    return y_k

def NARMA_Diverges(u):
    """
    Determine if random input creates a diverging NARMA10 sequence
    
    Args:
        u: Input data
        
    Returns:
        boolean: True if series diverges, false otherwise
    """
    
    #Generate first ten entries in series
    y_k = np.ones(10)*0.1
    length = len(u)

    #Iteratively calculate based on NARMA10 formula
    for k in range(10,length):
        t = 0.3*y_k[k-1]+0.05*y_k[k-1]*sum(y_k[k-1-i] for i in range(10))+1.5*\
        u[k-1]*u[k-10]+0.1
        
        #Series diverges if element greater than 1
        if t > 1:
            return True
        y_k = np.append(y_k,t)

    return False
    

def cross_validate(alphas,x,x_test,target):
    """
    Manual corss-validation, ie choosing optimal ridge parameter
    
    Args:
        alphas: ridge parameters to validate
        x: training data
        x_test: validation data
        target: correct labels for training/validation
        
    Returns:
        best_nrmse: lowest validation NRMSE found
        best_prediction: prediction with lowest validation NRMSE
        best_input: training prediction with lowest validation NRMSE
    """
    best_prediction = np.array([])
    best_nrmse = np.inf
    best_train = np.array([])
    np.append(alphas,0)
    for a in alphas:
        clf = Ridge(alpha = a)
        clf.fit(x,target[:len(x)])
        y_test = clf.predict(x_test)
        y_input = clf.predict(x)
        NRMSE = np.sqrt(np.mean(np.square(y_test[50:]-target[len(x)+50:]))/\
            np.var(target[len(x)+50:]))
        
        #Compare with previous best and update if better
        if(NRMSE < best_nrmse):
            best_nrmse = NRMSE
            best_prediction = y_test
            best_train = y_input
    
    return best_nrmse,best_prediction,best_train
        
    

def NARMA_Test(test_length = 800,train_length = 800,
        plot = True,N = 400,eta = 0.4,gamma = 0.05,tau = 400, fudge = 1,
        bits = np.inf, preload = False, write = False,mask = 0.1, activate = 'mg',
        cv = False,beta = 1.0,t = 1,theta = 0.2):
    """
    Args:
        test_length: length of testing data
        train_length: length of training data
        a: ridge regression parameter
        N: number of virtual nodes
        plot: display calculated time series
        gamma: input gain
        eta: oscillation strength
        bits: bit precision
        preload: preload mask and time-series data
        mask: amplitude of mask values
        activate: activation function to be used (sin**2,tanh,mg)
        cv: perform leave-one-out cross validation
        beta: driver gain
        t: timestep used to solve diffeq

    Returns:
        NRMSE: Normalized Root Mean Square Error
    """

    #Import u and m
    if preload:
        file1 = open("Data/Input_sequence.txt","r")
        file2 = open("Data/mask_2.txt","r")
        contents = file1.readlines()
        contents2 = file2.readlines()
        u = []
        m = []
        for i in range(1000):
            u.append(float(contents[i][0:contents[i].find("\t")]))
            if i < 400:
                m.append(float(contents2[i][0:contents2[i].find("\n")]))
        file1.close()
        file2.close()
        u = np.array(u)
        m = np.array(m)
        m = m[:N]           # Able to do preloaded data for all sorts of node sizes
    #Randomly initialize u and m
    else:
        u = np.random.rand(train_length+test_length)/2.
        while NARMA_Diverges(u):
            u = np.random.rand(train_length+test_length)/2.
        m = np.array([random.choice([-mask,mask]) for i in range(N)])
 
    #Calculate NARMA10 target
    target = NARMA_Generator(len(u),u)
    alphas = np.logspace(-20,5,16)
    
    #Instantiate Reservoir, feed in training and predictiondatasets
    r1 = DelayReservoir(N = N,eta = eta,gamma = gamma,theta = theta,
        beta = beta,tau = tau, fudge = fudge)  
    x = r1.calculate(u[:train_length],m,bits,t,activate)[0]
    x_test = r1.calculate(u[train_length:],m,bits,t,activate)[1]

    x_test_bot = r1.calculate(u[train_length:],m,bits,t,activate)[1]

    #Train using Ridge Regression with hyperparameter tuning
    if(cv):
        NRMSE,y_test,y_input = cross_validate(alphas,x,x_test,target)
    else:
        clf = Ridge(alpha = 0)
        clf.fit(x,target[:train_length])
        y_test = clf.predict(x_test)
        y_input = clf.predict(x)
    
        #Calculate NRMSE of prediction data
        NRMSE = np.sqrt(np.mean(np.square(y_test[50:]-target[train_length+50:]))/\
            np.var(target[train_length+50:]))
    
    #Calculate NRMSE of training data
    NRMSEi = np.sqrt(np.mean(np.square(y_input-target[:train_length]))/\
            np.var(target[:train_length]))
    

    #Write to File
    if(write):
        x_total = np.concatenate((x,x_test))
        x_total = x_total.flatten(order='C')
        file1 = open("data/x_comparison .txt","w+")
        for i in range(400000):
            file1.write("%f"%x_total[i]+"\n")
        file1.close()
    

    
    #Plot predicted Time Series
    if plot:
        plt.figure(1)
        plt.plot(np.linspace(0,1e-3*train_length,N*train_length),x.flatten()[0:])
        plt.xlabel('time [ms]')
        plt.ylabel('x(t) [V]')
        plt.figure(2)
        plt.plot(y_test[50:],label='Prediction')
        plt.plot(target[train_length+50:],label='Target')
        plt.title('NRMSE = %f'%NRMSE)
        plt.legend()

        plt.figure(2)
        plt.plot(y_test[50:],label='Prediction')
        plt.plot(target[train_length+50:],label='Target')
        plt.title('NRMSE = %f'%NRMSE)
        plt.legend()

        plt.figure(3)
        plt.plot(np.linspace(0,x_test_bot.flatten().shape[0],x_test_bot.flatten().shape[0]), x_test_bot.flatten()[0:], label = "[beta * x(t) + gamma * j(t)] ^ 1")
        plt.plot(np.linspace(0,N*u.shape[0],u.shape[0]),u.flatten()[0:], label = "Input Narma Taks")
        plt.xlabel("cycle * nodes")
        plt.title("MG Denominator compared to NARMA Input")
        plt.legend()

        plt.figure(4)
        plt.plot(np.linspace(0,x_test_bot.flatten().shape[0],x_test_bot.flatten().shape[0]), x_test_bot.flatten()[0:], label = "[beta * x(t) + gamma * j(t)] ^ 1")
        plt.xlabel("cycle * Nodes")
        plt.title("MG Denominator : [beta * x(t) + gamma * j(t)] ^ 1")

        plt.show()
    
    return NRMSE, x_test_bot


def run_test(eta, max_Tau, gamma, theta = 0.2, activation = "hayes"):
    """
    Run Narma tests. Simplifies parameter input for ease of reading
    
    Args:
        activation : "wright", "mg", "hayes" (in the future)
        eta : term that multiplies the delayed portion
        maxTau : the maximum tau for some given k (found through matlab program)
        theta : time spacing between nodes
        
    Returns:
        output: NRMSE of Narma task
    """
    Nodes =int(max_Tau//theta)         # First pass set theta to 0.2, may have to experimentally find ideal theta for wright eq.
    plot_yn = True          # Plot or no plot, that is the question
    # T = 0.08            # Time normalization constant?

    output = NARMA_Test(
            test_length = 800,
            train_length = 800,
            gamma = gamma,          # Input gain
            plot = plot_yn,
            N = Nodes,
            eta =  eta,           # parameter in front of delayed term (for wright: k)
            bits = np.inf,
            preload = False,
            cv = True,
            fudge = 1,
            beta = 1,
            tau = Nodes,
            activate = activation,
            theta = 0.2 
                )

    return output

def wright_grid_search(max_Tau = 9.992, wright_k = -0.15, eta = 0.05, theta = 0.2, gamma = 0.05, parameter_searched = "theta", activation = "hayes"):
    """
    Runs a grid search manipulating theta for the Wright equation. max_tau is the longest tau possible at wright_k.
    Holds all else constant and varies theta. Next ones will have to vary gamma.

    args:
        wright_k: k term in the Wright equation
        max_Tau: longest value of tau at wright_k
        eta: holding eta fixed case, default value of eta
        parameter_searched = which parameter(s) would you like to do a grid search for? ("theta", "eta_gamma")

    output:
        if "theta":
            graph showing NRMSE on vertical vs choices of theta
        if "eta_gamma":
            heatmap showing NRMSE various combinations of k(eta in program) and gamma
    """

    if parameter_searched == "theta":
        theta_range = np.linspace(0.2,0.3,10)           # Generate theta range
        wright_k = eta          # redefinition for ease of understanding that eta serves the role of k in the wright equation
        output_vals = {}            # Dictionary to hold output values
        for num,i in enumerate(theta_range):
            
            output = run_test(activation = activation, theta = i, eta = wright_k, max_Tau = max_Tau, gamma = gamma)
            output_vals[theta_range[num]] = output          # Collect theta value and corresponding NRMSE in a dictionary entry
        
        # Unzip dictionary entries
        lists = sorted(output_vals.items())
        x, y = zip(*lists)

        # Plot data
        plt.plot(x,y)
        plt.xlabel('theta')
        plt.ylabel('NRMSE')
        plt.show()

    if parameter_searched == "eta_gamma":
        interps = 10         # How many combinations of k and gamma would you like?
        k_range = np.linspace(-0.20,-0.15,interps)
        gamma_range = np.linspace(0.00,0.05,interps)
        eg_output = np.zeros((interps,interps))

        for k_index, kr in enumerate(k_range):
            for g_index, gr in enumerate(gamma_range):

                output = run_test(activation = activation, theta = theta, eta = kr, max_Tau = max_Tau, gamma = gr)
                eg_output[k_index,g_index] = output         # eg_output fills left corner to right corner, \n and so on
        
        # plot data 
        ax = sns.heatmap(eg_output, cmap = 'coolwarm', xticklabels = np.round_(gamma_range,4) , yticklabels = np.round_(k_range,4))
        ax.set(xlabel = "gamma", ylabel = "k", title = "Wright simulation results for NARMA task")
        plt.show()

def mg_hayes_comp():
    # Set large parameters for calculate
    t = 1
    bits = np.inf
    train_length = 800


    # Import data

    file1 = open("Data/Input_sequence.txt","r")         # Reads input and masking files. stores them in u/m
    file2 = open("Data/mask_2.txt","r")
    contents = file1.readlines()
    contents2 = file2.readlines()
    u = []
    m = []
    for i in range(1000):
        u.append(float(contents[i][0:contents[i].find("\t")]))
        if i < 400:
            m.append(float(contents2[i][0:contents2[i].find("\n")]))
    file1.close()
    file2.close()
    u = np.array(u)
    m = np.array(m)

    # MG portion, collect output
    activate = 'mg'
    r1 = DelayReservoir(N = 400,eta = 1,gamma = 0.05,theta = 0.2,\
        beta = 1,tau = 400)
    x_mg, vn_mg = r1.calculate(u[:train_length],m,bits,t,activate)         # Takes the actual output
    # x_mg_vn = r1.calculate(u[:train_length], m, bits, t, activate)[1]

    # Hayes portion, collect output
    activate = 'hayes'
    x_hayes, vn_hayes = r1.calculate(u[:train_length], m, bits, t, activate)
    # x_hayes_vn = r1.calculate(u[:train_length], m, bits, t, activate)[1]

    # Flatten the values
    x_mg = x_mg.flatten()
    x_hayes = x_hayes.flatten()

    vn_mg = vn_mg.flatten()
    vn_hayes = vn_hayes.flatten()

    # Plot the data
    plt.figure(1)
    plt.plot(x_mg, label = "mackey-glass")
    plt.plot(x_hayes, label = "hayes")
    plt.xlabel("cycle * Nodes")
    plt.title("Raw Mg vs Hayes Ouputs with Same NARMA Input")
    plt.legend()

    plt.figure(2)
    plt.plot(vn_mg, label = "mackey-glass vn")
    plt.plot(vn_hayes, label = "hayes vn")
    plt.xlabel("cycle * Nodes")
    plt.title("Raw VN: Mg vs Hayes Ouputs with Same NARMA Input")
    plt.legend()

    plt.show()

def ml_test_hayes(param):
    """
    Sets up 'function' to minimize. 
    Declares variables to hyperopt.
    Returns the mean NRMSE of 3 NARMA10 tasks
    """
    gamma, eta, beta, theta, hayes_param = \
        param['gamma'], param['eta'], param['beta'], param['theta'] , param['hayes_param']

    N = 400            # Number of Nodes

    return np.mean([NARMA_Test(
    test_length = 1000,
    train_length = 3200,
    gamma = gamma,
    plot = False,
    N = N,
    eta = eta,
    bits = np.inf,
    preload = False,
    beta = beta,
    fudge = hayes_param,
    tau = N,
    activate = 'hayes',
    theta = theta
    ) for _ in range(3)])

def hyperopt_grad_hayes():
    """
    calls the fmin 
    """
    best = fmin(
        fn = ml_test_hayes, 
        space = {
            # 'x': hp.randint('x',800),
            'gamma': hp.uniform('gamma', 0.01, 3),
            'eta': hp.uniform('eta', 0, 1),
            # 'N': hp.randint('N',800),
            'theta': hp.uniform('theta', 0, 1),
            'beta': hp.uniform('beta', 0, 1),
            'hayes_param': hp.uniform('hayes_param', -1, 0)
        },
        algo=tpe.suggest,
        max_evals=100
    )

    print(best)

def ml_test_wright(param):
    """
    Sets up 'function' to minimize. 
    Declares variables to hyperopt.
    Returns the mean NRMSE of 3 NARMA10 tasks
    """
    gamma, k, N, theta, beta = \
        param['gamma'], param['k'], param['N'],\
            param['theta'], param['beta']

    return np.mean([NARMA_Test(
    test_length = 800,
    train_length = 3200,
    gamma = gamma,
    plot = False,
    N = N,
    eta = k,
    bits = np.inf,
    preload = False,
    beta = beta, 
    tau = N,
    activate = 'hayes',
    theta = theta
    ) for _ in range(3)])

def hyperopt_grad_wright():
    """
    calls the fmin 
    """
    best = fmin(
        fn = ml_test_hayes, 
        space = {
            # 'x': hp.randint('x',800),
            'gamma': hp.uniform('gamma', 0.01, 2),
            'eta': hp.uniform('k', 0, 1),
            'N': hp.randint('N',800),
            'theta': hp.uniform('theta', 0.0001, 1),
            'beta': hp.uniform('beta', 0, 1)
        },
        algo=tpe.suggest,
        max_evals=100
    )

    print(best)
    
def make_training_testing_set(num_sin=3000,num_saw=3000,num_square=3000, test_percent=0.1):
    sin = lambda x,theta: np.sin(theta*x)
    square = lambda x, theta: np.sign(theta*x)
    sawtooth = lambda x, theta: signal.sawtooth(theta*x)
    t = np.linspace(-5, 5, 500)
    sins = []
    saws = []
    squares = []
    for x in range(num_sin):
        # print(f'sin: {x}')
        theta = np.random.uniform(0.001,5)
        wave = sin(t, theta)
        for idx in range(len(wave)):
            noise = 0.02 * np.random.normal()
            if np.random.uniform(0,1) <= 0.5:
                wave[idx] += noise
            else:
                wave[idx] -= noise
        sins.append(wave)
    for x in range(num_saw):
        # print(f'saw: {x}')
        theta = np.random.uniform(0.001,5)
        wave = sawtooth(t, theta)
        for idx in range(len(wave)):
            noise = 0.02 * np.random.normal()
            if np.random.uniform(0,1) <= 0.5:
                wave[idx] += noise
            else:
                wave[idx] -= noise
        saws.append(wave)
    for x in range(num_square):
        # print(f'square: {x}')
        theta = np.random.uniform(0.001,5)
        wave = square(t, theta)
        for idx in range(len(wave)):
            noise = 0.02 * np.random.normal()
            if np.random.uniform(0,1) <= 0.5:
                wave[idx] += noise
            else:
                wave[idx] -= noise
        squares.append(wave)
    X = np.concatenate([sins,saws,squares])
    y = np.concatenate([[0 for _ in range(num_sin)], [1 for _ in range(num_saw)], [2 for _ in range(num_square)]])

    return train_test_split(X, y, test_size = test_percent, random_state = 42)

def Classification_Test(num_loops=1, N=400, eta=[0.35], gamma=[0.05], phi=[0.09 * np.pi], tau=[400],
                bits=np.inf, preload=False, write=False, mask=0.1, activate='mg',
                beta=[1.0], wright_param=-1,power=7, t=1):
     """
	 Args:
		 test_length: length of testing data
		 train_length: length of training data
		 num_loops: number of delay loops in reservoir
		 a: ridge regression parameter
		 N: number of virtual nodes
		 plot: display calculated time series
		 gamma: input gain
		 eta: oscillation strength
		 phi: phase of MZN
		 r: loop delay length
		 bits: bit precision
		 preload: preload mask and time-series data
		 mask: amplitude of mask values
		 activate: activation function to be used (sin**2,tanh,mg)
		 cv: perform leave-one-out cross validation
		 beta: driver gain
		 V_low: ADC lower bound
		 V_high: ADC upper bound
		 t: timestep used to solve diffeq
		 layers: number of hidden layers, ie number of cascaded reservoirs
		 sr: splitting ratio
		 switching: WDM switching
		 w: number of wavelengths
		 auto: automatically calculate ADC range
		 IA: input to all layers of (deep) network

	 Returns:
		 NRMSE: Normalized Root Mean Square Error
	 """

     X_train, X_test, y_train, y_test = make_training_testing_set(num_sin=3333,num_saw=3333,num_square=3334,test_percent=0.1)

     clf = RidgeClassifier(alpha=0)
     m = np.array([random.choice([-mask, mask]) for i in range(N)])

     # Instantiate Reservoir, feed in training and predictiondatasets
     r1 = DelayReservoir(N=N, eta=eta, gamma=gamma, theta=0.2,
                         loops=num_loops, phi=phi, beta=beta, tau=tau, wright_param=wright_param, power=power)
     Xs = []
     for idx in tqdm(len(X_train)):
        Xs.append(np.array(r1.calculate(X_train[idx], m, bits, t, activate, V_low=0.0, V_high=1.0, sr=0.5)).flatten())
     Xs = np.array(Xs)
     clf.fit(Xs,y_train)
     Xs = []
     for idx in tqdm(len(X_test)):
        print(idx)
        Xs.append(np.array(r1.calculate(X_test[idx], m, bits, t, activate, V_low=0.0, V_high=1.0, sr=0.5)).flatten())
     Xs = np.array(Xs)

     return clf.score(Xs,y_test)




##### TESTS #####

#### mg_hayes_comp tests ####
mg_hayes_comp()

#### Grid search tests ####
# wright_grid_search(parameter_searched="eta_gamma", activation = "hayes")


#### run_test tests ####
# run_test(eta = 0.5, max_Tau = 9.993, gamma = 0.7, theta = 0.2, activation = "wright")
# run_test(eta = 0.05, max_Tau = 150, gamma = 0.5, theta = 0.2, activation = "hayes")


#### Hyperopt Tests ####
# hyperopt_grad_wright()
# hyperopt_grad_hayes()


#### General Test for NARMA_Test ####
# NARMA_Test(
#     test_length = 800,
#     train_length = 3200,
#     gamma = 0.48707341674880045,
#     plot = False,
#     N = 317,
#     eta = 0.9615229252495553
#     bits = np.inf,
#     preload = False,
#     beta = 0.408835328209339, 
#     tau = x,
#     activate = 'hayes',
#     theta = theta
#     )