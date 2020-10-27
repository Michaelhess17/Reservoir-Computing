#!/usr/bin/env python3

__author__ = "Philip Jacobson"
__email__ = "philip_jacobson@berkeley.edu"

import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from Delay_Reservoir import DelayReservoir
from helper_files import NARMA_Generator, cross_validate, NARMA_Diverges


# from hyperopt import hp, tpe, fmin


############################################################################

class ModifiedDelayRC(DelayReservoir):
	def __init__(self, eta=0.35, gamma=0.05, beta=1.0, theta=0.2, tau=400, N=400, power=1):
		super().__init__(eta=0.35, gamma=0.05, beta=1.0, theta=0.2, tau=400, N=400, power=1)
		self.eta = eta
		self.gamma = gamma
		self.beta = beta
		self.tau = tau
		self.N = N
		self.power = power

	def calculate(self, u, m, t, act, noise_scale):
		"""
		Calculate reservoir state over duration u

		Args:
			u: input data
			m: mask array
			bits: number of bit precision, np.inf for analog values
			noise_scale: noise amplitude in input
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
		M_x = np.zeros(J.shape)

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
		for i in range(1, (cycles + 1) * self.N * t // 1):
			M_x[0, i] += noise_scale*np.random.uniform(-0.5, 0.5)

		M_x = M_x.reshape(1 + cycles, self.N)

		return M_x[1:,:]

def NARMA_Test(test_length=800, train_length=800, N=400, eta=0.4, gamma=0.05,
               tau=400, bits=np.inf, preload=False, mask=0.1, activate='mg', cv=True, beta=1.0, t=1,
               theta=0.2, scale=True, noise_scale=0):
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
	# Import u and m
	if preload:
		file1 = open("Data/Input_sequence.txt", "r")  # Reads input and masking files. stores them in u/m
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
	# Randomly initialize u and m
	else:
		u = np.random.rand(train_length + test_length) / 2.
		while NARMA_Diverges(u):
			u = np.random.rand(train_length + test_length) / 2.
		m = np.array([random.choice([-mask, mask]) for i in range(N)])

	# Calculate NARMA10 target
	target = NARMA_Generator(len(u), u)
	alphas = np.logspace(-20, 5, 16)

	# Instantiate Reservoir, feed in training and predictiondatasets
	r1 = ModifiedDelayRC(N=N, eta=eta, gamma=gamma, theta=theta,
	                    beta=beta, tau=tau)
	scaler = StandardScaler()
	x = r1.calculate(u[:train_length], m, t, activate, noise_scale)
	x_test = r1.calculate(u[train_length:], m, t, activate, noise_scale)
	if scale:
		scaler.fit(x)
		x = scaler.transform(x)
		x_test = scaler.transform(x_test)
		# for idx in range(x.shape[1]):
		# 	scaler.fit(x[:, idx].reshape(-1, 1))
		# 	x[:, idx] = scaler.transform(x[:, idx].reshape(-1, 1)).reshape(800)
		# 	x_test[:, idx] = scaler.transform(x_test[:, idx].reshape(-1, 1)).reshape(800)
	# Train using Ridge Regression with hyperparameter tuning
	if (cv):
		NRMSE, y_test, y_input, clf = cross_validate(alphas, x, x_test, target)
	else:
		clf = Ridge(alpha=0)

		clf.fit(x, target[:train_length])
		y_test = clf.predict(x_test)
		y_input = clf.predict(x)

		# Calculate NRMSE of prediction data
		NRMSE = np.sqrt(np.mean(np.square(y_test[50:] - target[train_length + 50:])) / \
		                np.var(target[train_length + 50:]))

	return NRMSE


def Mask_Test():
	"""
	Args:
		test_length: length of testing data
		train_length: length of training data
		a: ridge regression parameter
		N: number of virtual high_nodes
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
	masks = np.logspace(-4, 1, 30)
	acts = ['mg', 'hayes']
	colors = ['r', 'b']

	for k, act in enumerate(acts):
		NRMSEs = []
		for mask in masks:
			NRMSEs.append(np.mean([NARMA_Test(activate=act, mask=mask) for _ in range(3)]).item())
		plt.semilogx(masks, NRMSEs, c=colors[k], label=act)
	plt.xlabel('Scale of Mask')
	plt.ylabel('NRMSE')
	plt.legend()
	plt.savefig(f'Masking Tolerance.png', dpi=600)

Mask_Test()


