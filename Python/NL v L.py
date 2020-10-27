import random

import numpy as np
from sklearn.linear_model import Ridge

from Delay_Reservoir import DelayReservoir
from helper_files import load_NARMA, cross_validate


class ModifiedDelayRC(DelayReservoir):
	def __init__(self, eta=0.9, gamma=0.0, beta=1.0, theta=0.2, tau=400, N=400, power=7, fudge=1.0):
		DelayReservoir.__init__(self, eta=0.9, gamma=0.0, beta=1.0, theta=0.2, tau=400, N=400, power=7, fudge=fudge)
		self.eta = eta
		self.gamma = gamma
		self.beta = beta
		self.tau = tau
		self.N = N
		self.power = power
		self.fudge = fudge

	def calculate_mix(self, u, m, t, mix_p):
		# Reshape input data with mask

		J = self.mask(u, m)
		cycles = J.shape[0]

		# Add extra layer to match indexes with M_x
		J = np.vstack((np.zeros((1, self.N)), J))
		J = J.flatten(order='C')
		J = J.reshape((1, (1 + cycles) * self.N), order='F')
		M_x = np.zeros(J.shape)
		X_lag = np.zeros(J.shape)
		acts = random.choices(['mg', 'hayes'], [1-mix_p, mix_p], k=self.N)
		# Iteratively solve differential equation with Euler's Method
		for i in range(1, (cycles + 1) * self.N * t // 1):  # // 1 is probably useless but i am afraid to take it out
			a = acts[(i-1) % self.N]
			a = self.activationFunction(a)
			vn = M_x[0, i - 1] - self.fudge * M_x[0, i - 1] * self.theta / t
			arg = M_x[0, i - 1 - self.tau * t]
			X_lag[0, i] = arg
			vn += self.eta * a(self.beta * arg + self.gamma * J[0, (i - 1) // t]) * self.theta / t
			M_x[0, i] = vn
		# Before M_x[i].reshape(1+cycles,self.N)

		M_x = M_x.reshape(1 + cycles, self.N)
		M_x = M_x[1:, :]
		X_lag = X_lag.reshape(1 + cycles, self.N)
		J = J.reshape(1 + cycles, self.N)

		return M_x, J, X_lag

	def NARMA_Test(self, test_length=500, train_length=5000,
	               plot=False, N=400, eta=0.4, gamma=0.05, tau=400, fudge=1.0,
	               preload=False, write=False, mask=0.1, activate='mg',
	               cv=False, beta=1.0, t=1, theta=0.2, power=1, mix_p=0.1):
		"""
		Args:
			test_length: length of testing data
			train_length: length of training data
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
			t: timestep used to solve diffeq,
			theta: distance between virtual high_nodes in time

		Returns:
			NRMSE: Normalized Root Mean Square Error
		"""

		# Import u, m, and target
		u, m, target = load_NARMA(preload, train_length, test_length, mask, N)

		# Instantiate Reservoir, feed in training and predictiondatasets
		r1 = ModifiedDelayRC(N=N, eta=eta, gamma=gamma, theta=theta,
		                     beta=beta, tau=tau, fudge=fudge, power=power)
		x = r1.calculate_mix(u[:train_length], m, t, mix_p)[0]
		# Is this correct? It looks like x_test and x_test_bot are defined as the same thing
		x_test = r1.calculate_mix(u[train_length:], m, t, mix_p)[0]

		# Train using Ridge Regression with hyperparameter tuning
		if cv:
			alphas = np.logspace(-100, 1, 100)
			NRMSE, y_test, y_input1, clf = cross_validate(alphas=alphas, x=x, x_test=x_test, target=target)

		else:
			clf = Ridge(alpha=0)
			# clf1 = LinearRegression(n_jobs=-1)
			clf.fit(x, target[:train_length])
			y_test = clf.predict(x)

			# Calculate NRMSE of prediction data
			NRMSE = np.sqrt(
				np.mean(np.square(y_test[50:] - target[50:train_length])) / np.var(target[50:train_length]))

		return NRMSE, x, target, x_test, y_test, clf

import matplotlib.pyplot as plt

MR = ModifiedDelayRC()
NRMSE = np.zeros(20)
for k, mix_p in enumerate(np.linspace(0, 1, 20)):
	NRMSE[k] = MR.NARMA_Test()[0]

plt.plot(NRMSE)
plt.show()
