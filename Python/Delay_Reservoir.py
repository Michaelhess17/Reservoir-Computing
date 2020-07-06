#!/usr/bin/env python3

__author__ = "Philip Jacobson"
__email__ = "philip_jacobson@berkeley.edu"

import random

import numpy as np


##########################################################################

class DelayReservoir():
    """
    Class to perform Reservoir Computing using Delay-Based Architecture
    """
    
    def __init__(self, N = 400, eta = 0.4,gamma = 0.05,theta = 0.2,beta = 1.0,tau = 400, fudge = 1, power = 1):
        """
        Args:
            N:  Number of Virtual Nodes
            eta: Feedback gain
            gamma: Input gain
            theta: Distance between virtual nodes
            beta: Driver gain 
            tau: Ratio of loop length to node spacing for each loop
        """
        
        self.N = N
        self.eta = eta
        self.gamma = gamma
        self.theta = theta
        self.beta = beta
        self.tau = tau
        self.fudge = fudge
        self.power = power

    def mask(self,u,m = None):
        """
        Args:
            u: Input data
            m: Mask array

        Returns:
            J: Multiplexed (masked) data
        """

            
        if len(m.shape) == 1:
            
            if m.all() == None:
                m = np.array([random.choice([-0.1,0.1]) for i in range(self.N)])
            
            u = np.reshape(u,(-1,1))
            m = np.reshape(m,(1,-1))
            
            return u@m
        else:
            return (m@u).T

    def calculate(self,u,m,bits,t,act,no_act_res = False):
        """
        Calculate reservoir state over duration u

        Args:
            u: input data
            m: mask array
            bits: number of bit precision, np.inf for analog values
            noise: noise amplitude in input
            t: ratio of node interval to solver timestep
            act: activation function to be used for nonlinear node
            no_act_res: Return, in addition to regular solution, the base vn for each series

        Returns:
            M_x: matrix of reservoir history
        """
        
        
        #Reshape input data with mask

        J = self.mask(u,m)
        cycles = J.shape[0]
            
        #Add extra layer to match indexes with M_x
        J = np.vstack((np.zeros((1,self.N)),J))
        J = J.flatten(order= 'C')
        J = J.reshape((1,(1+cycles)*self.N),order = 'F')
        M_x = np.zeros(J.shape)
        Mx_no_act = np.zeros(J.shape)         # Create container to store values of denominator 
    
        
        #Select activation function
        a = self.activationFunction(act)

        #Iteratively solve differential equation with Euler's Method  
        for i in range(1,(cycles+1)*self.N*t//1): 
            vn = M_x[0,i-1]-M_x[0,i-1]*self.theta/t
            
            if act == "wright":         # In the case that we want wright, take out the x(t) term by redefining vn
                vn = M_x[0,i-1]

            arg = M_x[0,i-1-self.tau*t] 
            vn += self.eta*a(self.beta*arg+self.gamma*J[0,(i-1)//t]) * self.theta/t
            M_x[0,i] = vn
            
        #Reshape matrix
        M_x_new = np.zeros((1+cycles,self.N*t))

        # M_x_new[:,i*self.N:(i+1)*self.N] = \
        #     M_x[0,i].reshape(1+cycles,self.N)         # Before M_x[i].reshape(1+cycles,self.N)  

        M_x_new = M_x.reshape(1+cycles,self.N)


        if no_act_res:

            # Loop through and solve no_activation portions
            for i in range(1,(cycles+1)*self.N*t//1): 
                vn_no_act = Mx_no_act[0,i-1]-M_x[0,i-1]*self.theta/t
                vn_no_act += self.eta*(self.beta*arg+self.gamma*J[0,(i-1)//t]) * self.theta/t
                Mx_no_act[0,i] = vn_no_act  # Store the denominator values (without the "1+", this can be added in if needed for computation)

            # Reshape Matrix
            Mx_new_nAct= np.zeros((1+cycles, self.N*t))  
            Mx_new_nAct = Mx_new_nAct.reshape(1+cycles,self.N)

            return M_x_new[1:,0:self.N*t:t], Mx_new_nAct[1:,0:self.N*t:t]  


        #Remove first row of zeroes, select values at node spacing
        return M_x_new[1:,0:self.N*t:t]

    def activationFunction(self,func):
        """
        Choose and evaluate correspinding activation function (tanh,sin^2,
        mackey-glass)
        
        args:
            func: activation function type, either 'tanh', 'sin', or 'mg'
            
        Returns:
            a: lambda function acting as activation function
        """
        
        #Return correct function, otherwise raise an exception
        if(func == 'mg'):
            return lambda x: x/((1+x)**self.power)
        elif (func == 'wright'):
            return lambda x: x 
        elif (func == 'hayes'):
            return lambda x: x
        elif (func == 'mod_hayes'):
            return lambda x: x/(1+round(float(np.random.rand(1)), 3))
        else:
            raise Exception('Not a valid activation function!')