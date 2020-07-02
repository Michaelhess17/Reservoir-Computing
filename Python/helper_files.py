
import numpy as np
import random
import multiprocessing
import csv
import time
import pandas
from Delay_Reservoir import DelayReservoir
from joblib import Parallel, delayed

from hyperopt import fmin, tpe, hp
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
            best_nrmse = NRMSE
            best_prediction = y_test
            best_train = y_input
    
    return best_nrmse,best_prediction,best_train
        
    
