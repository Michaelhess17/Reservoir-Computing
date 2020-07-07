#!/usr/bin/env python3

__author__ = "Philip Jacobson"
__email__ = "philip_jacobson@berkeley.edu"

import numpy as np
from Delay_Reservoir import DelayReservoir
from helper_files import load_NARMA


# from hyperopt import hp, tpe, fmin


############################################################################

class ModifiedDelayRC(DelayReservoir):
	def __init__(self, eta=0.9, gamma=0, beta=1.0, theta=0.2, tau=400, N=400, power=7):
		super().__init__(eta=0.9, gamma=0, beta=1.0, theta=0.2, tau=400, N=400, power=7)
		self.eta = eta
		self.gamma = gamma
		self.beta = beta
		self.tau = tau
		self.N = N
		self.power = power

	def calculate(self, u, m, t, act):
		"""
		Calculate reservoir state over duration u

		Args:
			u: input data
			m: mask array
			bits: number of bit precision, np.inf for analog values
			noise: noise amplitude in input
			t: ratio of node interval to solver timestep
			act: activation function to be used for nonlinear node

		Returns:
			M_x: matrix of reservoir history
		"""

		# Reshape input data with mask

		J = self.mask(u, m)
		cycles = J.shape[0]

		# Add extra layer to match indexes with M_x
		J = np.vstack((np.zeros((1, self.N)), J))
		J = J.flatten(order='C')
		J = J.reshape((1, (1 + cycles) * self.N), order='F')
		M_x = 1/2*np.ones(J.shape)

		# Select activation function
		a = self.activationFunction(act)

		# Iteratively solve differential equation with Euler's Method
		for i in range(1, (cycles + 1) * self.N * t // 1):
			vn = M_x[0, i - 1] - M_x[0, i - 1] * self.theta / t

			if act == "wright":  # In the case that we want wright, take out the x(t) term by redefining vn
				vn = M_x[0, i - 1]

			arg = M_x[0, i - 1 - self.tau * t]
			vn += self.eta * a(self.beta * arg + self.gamma * J[0, (i - 1) // t]) * self.theta / t
			M_x[0, i] = vn
		# Before M_x[i].reshape(1+cycles,self.N)

		M_x = M_x.reshape(1 + cycles, self.N)

		return M_x


def Bif_Test(test_length=800, train_length=800, N=400, eta=0.4,  tau=400,
			   bits=np.inf, preload=False, write=False, mask=0.1, activate='mg',  beta=1.0, t=1, theta=0.2, hayes_p=1,power=1):
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

	# Import u and m
	u, m, _ = load_NARMA(preload, train_length, test_length, mask, N)

	# Instantiate Reservoir, feed in training and predictiondatasets
	r1 = DelayReservoir(N=N, eta=eta, gamma=0, theta=theta,
						beta=beta, tau=tau, fudge=hayes_p, power = power)
	x = r1.calculate(u[:train_length], m, bits, t, activate)
	x_max = np.max(x)
	x_min = np.min(x)
	return x_min, x_max


def run_test():
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
	a = 8000
	output = np.zeros((2,a))
	for tau in range(3000, a):
		print(tau)
		mini, maxi = Bif_Test(
			test_length=800,
			train_length=800,
			N=400,
			eta=0.9,  # parameter in front of delayed term (for wright: k)
			bits=np.inf,
			preload=True,
			beta=1,
			tau=tau+1,
			activate='mg',
			theta=0.2,
			power = 7
		)
		output[0,tau] = mini
		output[1,tau] = maxi
	return output

import matplotlib.pyplot as plt

# out = run_test()
# plt.scatter(range(len(out[0,:])), out[0,:])
# plt.scatter(range(len(out[0,:])), out[1,:])
# plt.show


preload = False
train_length, test_length = 800, 800
mask = 0.1
N, tau = 400, 200000
eta, theta, beta = 1.0, 0.2, 0.5
power = 7
activate = 'mg'
t = 1
gamma = 0.5


u, m, _ = load_NARMA(preload, train_length, test_length, mask, N)
r1 = ModifiedDelayRC(N=N, eta=eta, gamma=gamma, theta=theta, beta=beta, tau=tau, power=power)
x = r1.calculate(u[:train_length], m, t, activate)
print(np.max(x[len(x)//2:]))
plt.plot(x[len(x)//2:])
plt.show()
