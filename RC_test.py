#!/usr/bin/env python3

import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from Delay_Reservoir import DelayReservoir

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


def NARMA_Test(test_length = 800,train_length = 800,num_loops = 1,a = 0,
        plot = True,N = 400,eta = 0.2,gamma = 0.05,phi = np.pi/6,tau = 400,
        bits = 8,preload = False):
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

    Returns:
        NRMSE: Normalized Root Mean Square Error
    """
    
    #Import u and m
    if preload:
        file1 = open("data/Input_sequence.txt","r")
        file2 = open("data/mask_2.txt","r")
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
    #Randomly initialize u and m
    else:
        u = np.random.rand(train_length+test_length)/2.
        m = np.array([random.choice([-0.1,0.1]) for i in range(N//num_loops)])
 
    #Calculate NARMA10 target
    target = NARMA_Generator(len(u),u)
    
    #Instantiate Reservoir, feed in training and verification datasets
    r1 = DelayReservoir(N = N//num_loops,eta = eta,gamma = gamma,theta = 0.2,
        loops=num_loops,phi = phi)
    x = r1.calculate(u[:train_length],m,bits)
    #x_ideal = r1.calculateMZN(u[:train_length],m)
    x_test = r1.calculate(u[train_length:],m,bits)
    #x_test_ideal = r1.calculateMZN(u[train_length:],m) 
    
    #Train using Ridge Regression
    #clf = RidgeCV(alphas = a,fit_intercept = True)
    clf = Ridge(alpha = a)
    clf.fit(x,target[:train_length])
    y_test = clf.predict(x_test)
    y_input = clf.predict(x)

    #Calculate NRMSE
    NRMSE = np.sqrt(np.mean(np.square(y_test[50:]-target[train_length+50:]))/\
            np.var(target[train_length+50:]))
    
    NRMSEi = np.sqrt(np.mean(np.square(y_input-target[:train_length]))/\
            np.var(target[:train_length]))
    
    #Write to File
    '''
    x_total = np.concatenate((x,x_test))
    x_total = x_total.flatten(order='C')
    file1 = open("data/64_bit_test_x.txt","w+")
    file2 = open("data/64_bit_test_y.txt","w+")
    for i in range(2*320000):
        file1.write("%f"%x_total[i]+"\n")
        if(i < 1600):
            file2.write("%f"%target[i]+"\n")
    file1.close()
    '''

    
    #Plot predicted Time Series
    if(plot == True):
        #fig, (ax1,ax2) = plt.subplots(2,1)
        #ax1.plot(x.flatten()[5000:])
        #ax2.plot(x_ideal.flatten()[5000:])
        #plt.plot(x.flatten()[:1200])
        plt.plot(y_test[50:],label='Prediction')
        plt.plot(target[train_length+50:],label='Target')
        plt.title('NRMSE = %f'%NRMSE)
        plt.legend()
        plt.show()

    return NRMSE

def NARMA_Test_Compare(test_length = 200,train_length = 800,num_loops = 1,
        a = 0,plot = True,N = 400,eta = 0.5,gamma = 1,phi = np.pi/4,r = 1):
    """
    Compare with pre-determined NARMA10 series

    Args:
        test_length: length of verification data
        train_length: length of training data
        num_loops: number of delay loops in reservoir
        a: list of ridge regression constants for hyperparameter tuning
        N: number of virtual nodes
        plot: display calculated time series
        gamma: input gain
        eta: oscillation strength
        phi: phase of MZN
        r: loop length ratio

    Returns:
        NRMSE: Normalized Root Mean Square Error
    """
    
    #Import u and m
    file1 = open("data/uin_and_target.txt","r")
    file2 = open("data/Mask.txt","r")
    contents = file1.readlines()
    contents2 = file2.readlines()
    u = []
    target = []
    m = []
    for i in range(1000):
        u.append(float(contents[i][0:contents[i].find("\t")]))
        target.append(float(contents[i][contents[i].find("\t"):]))
        if i < 400:
            m.append(float(contents2[i][0:contents2[i].find("\n")]))
    file1.close()
    file2.close()
    u = np.array(u)
    m = np.array(m)
    target = np.array(target)
    
    #Instantiate Reservoir, feed in training and verification datasets
    r1 = DelayReservoir(N = N//num_loops,eta = eta,gamma = gamma,theta = 0.2,
        loops=num_loops,phi = phi)
    x = r1.calculate(u[:train_length],m)
    x_test = r1.calculate(u[train_length:],m)
    
    x = []
    file3 = open("data/X_node.txt","r")
    contents3 = file3.readlines()
    print(len(contents3))
    for i in range(400000):
        x.append(float(contents3[i][:contents3[i].find("\n")]))
    
    x = np.array(x)
    x = x.reshape((-1,1))
    x = x.reshape((1000,400))

    #Train using Ridge Regression
    clf = Ridge(alpha = a,fit_intercept=True)
    clf.fit(x[:800],target[:train_length])
    w = clf.coef_
    y_train = x@w
    y_test = clf.predict(x[800:])
    
    #Write to file

    x_total = np.concatenate((x,x_test))
    x_total = x_total.flatten(order='C')
    file3 = open("data/y_train2.txt","w+")
    for i in range(800):
        file3.write("%f"%y_train[i]+"\n")
    file3.close()

    
    #Calculate NRMSE
    NRMSE = np.sqrt(np.mean(np.square(y_test[50:]-target[train_length+50:]))/\
            np.var(target[train_length+50:]))
    
    #Plot predicted Time Series
    
    if(plot == True):
        plt.plot(y_test[50:],label='Prediction')
        plt.plot(target[train_length+50:],label='Target')
        plt.title('NRMSE = %f'%NRMSE)
        plt.legend()
        plt.show()
    
    return NRMSE


#alphas = np.logspace(-10,-6,100)
print(NARMA_Test(test_length = 800,train_length = 800,num_loops = 1,
    a = 1e-8,gamma = 0.05,plot = True,N = 400,eta = 0.4,tau = 400,bits = 20,
    preload = False))


#print(NARMA_Test_Compare())


'''
l = np.linspace(1e-11,1e-10,20)
error1 = []
error2 = []
for i in range(10):
    #error1.append(NARMA_Test(test_length = 800,train_length = 800,
    #    num_loops = 1,a = 0, plot = False,N = 200))
    error2.append(NARMA_Test(test_length = 800,train_length = 800,
        num_loops = 1,a = 5e-15,gamma = 0.05, plot = False,N = 400,bits = 20))




#plt.plot(10*np.log10(g[1:]),error2[1:])
#plt.ylabel('NRMSE')
#plt.xlabel('Input gain [dB]')
#plt.show()
#print(np.mean(error1),np.std(error1))
#print(error2)
print(np.mean(error2),np.std(error2))
'''
