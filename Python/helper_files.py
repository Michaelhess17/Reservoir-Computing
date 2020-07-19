
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# from hyperopt import hp, tpe, fmin

"""
Utility Functions Used by RC_test.py
"""
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
            optimal_alpha = a
            best_nrmse = NRMSE
            best_prediction = y_test
            best_train = y_input
    
    return best_nrmse,best_prediction,best_train


def make_training_testing_set(num_waves=1000, test_percent=0.1, preload=False, write=False):
    if not preload:
        num_sin, num_saw, num_square = [num_waves // 3]*3
        sin = lambda x,theta: np.sin(theta*x)
        square = lambda x, theta: np.sign(theta*x)
        sawtooth = lambda x, theta: signal.sawtooth(theta*x)
        t = np.linspace(-5, 5, 500)
        waves = []
        y = []
        funcs = [sin, sawtooth, square]
        totals = [num_sin, num_saw, num_square]
        for i in range(len(funcs)):
            func = funcs[i]
            for x in range(totals[i]):
                y.append(i)
                theta = np.random.uniform(0.001,5)
                wave = func(t, theta)
                for idx in range(len(wave)):
                    noise = 0.02 * np.random.normal()
                    if np.random.uniform(0,1) <= 0.5:
                        wave[idx] += noise
                    else:
                        wave[idx] -= noise
                waves.append(wave)
        X = np.array(waves)
        y = np.array(y)

        if write:
            x_total = np.concatenate((X,y))
            # x_total = x_total.flatten(order='C')
            file = open("data/sin_waves.txt", "w+")
            for i in range(len(x_total)):
                file.write("%f" % x_total[i] + "\n")
            file.close()

        return train_test_split(X, y, test_size = test_percent, random_state = 42)


def load_NARMA(preload, train_length=800, test_length=800, mask=0.1, N=400):
    if preload:
        file1 = open("Data/Input_sequence.txt", "r")
        file2 = open("Data/mask_2.txt", "r")
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
        m = m[:N]  # Able to do preloaded data for all sorts of node sizes

        if N > 400:         # If you want more than 400 nodes:
            np.random.seed(10)          # Resets the seed so behaves the same
            m = np.random.choice([0.1,-0.1], [1,N])         # Tailor the mask to the number of nodes
            m = m.reshape(N,)

    # Randomly initialize u and m
    else:
        u = np.random.rand(train_length + test_length) / 2.
        while NARMA_Diverges(u):
            u = np.random.rand(train_length + test_length) / 2.
        m = np.array([random.choice([-mask, mask]) for _ in range(N)])

    target = NARMA_Generator(len(u), u)
    return u, m, target


def plot_func(x, x_test_bot, u, y_test, target, NRMSE, train_length, N):
    if x_test_bot == 0:
        plt.figure(1)
        plt.plot(np.linspace(0, 1e-3 * train_length, N * train_length), x.flatten()[0:])
        plt.xlabel('time [ms]')
        plt.ylabel('x(t) [V]')

        plt.figure(2)
        plt.plot(y_test[50:], label='Prediction')
        plt.plot(target[train_length + 50:], label='Target')
        plt.title('NRMSE = %f' % NRMSE)

        # plt.xlabel('N =400, eta = 0.75, gamma = 0.05, tau = 400, beta =1,theta = 0.2, k1 = 1, act = mg')         # Must be Changed Manually....
        
        plt.legend()

    else:
        plt.figure(3)
        plt.plot(np.linspace(0, x_test_bot.flatten().shape[0], x_test_bot.flatten().shape[0]), x_test_bot.flatten()[0:],
                label="[beta * x(t) + gamma * j(t)] ^ 1")
        plt.plot(np.linspace(0, N * u.shape[0], u.shape[0]), u.flatten()[0:], label="Input Narma Taks")
        plt.xlabel("cycle * nodes")
        plt.title("MG Denominator compared to NARMA Input")
        plt.legend()

        plt.figure(4)
        plt.plot(np.linspace(0, x_test_bot.flatten().shape[0], x_test_bot.flatten().shape[0]), x_test_bot.flatten()[0:],
                label="[beta * x(t) + gamma * j(t)] ^ 1")
        plt.xlabel("cycle * Nodes")
        plt.title("MG Denominator : [beta * x(t) + gamma * j(t)] ^ 1")

    plt.show()


def write_func(x, x_test):
    x_total = np.concatenate((x, x_test))
    x_total = x_total.flatten(order='C')
    file1 = open("data/x_comparison .txt", "w+")
    for i in range(400000):
        file1.write("%f" % x_total[i] + "\n")
    file1.close()

import heapq

def margin(model, X):
    preds = model.decision_function(X)
    margins = []
    for idx in range(len(preds)):
        top, next = heapq.nlargest(2, preds[idx])
        margins.append(top-next)
    return margins



