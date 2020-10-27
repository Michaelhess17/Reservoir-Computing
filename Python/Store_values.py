import numpy as np
from sklearn.linear_model import Ridge

from Delay_Reservoir import DelayReservoir
from helper_files import load_NARMA, cross_validate


# @ignore_warnings(category=ConvergenceWarning)
# def cross_validate(l1_ratios, x, x_test, target):
# 	"""
# 	Manual corss-validation, ie choosing optimal ridge parameter
#
# 	Args:
# 		alphas: ridge parameters to validate
# 		x: training data
# 		x_test: validation data
# 		target: correct labels for training/validation
#
# 	Returns:
# 		best_nrmse: lowest validation NRMSE found
# 		best_prediction: prediction with lowest validation NRMSE
# 		best_input: training prediction with lowest validation NRMSE
# 	"""
# 	clf = ElasticNetCV(n_alphas=1000, l1_ratio=l1_ratios, n_jobs=-1)
# 	clf.fit(x, target[:len(x)])
# 	y_test = clf.predict(x_test)
# 	y_input = clf.predict(x)
# 	NRMSE = np.sqrt(np.mean(np.square(y_test[50:] - target[len(x) + 50:])) / np.var(target[len(x) + 50:]))
#
# 	return NRMSE, y_test, y_input, clf


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

	def calculate(self, u, m, t, act, no_act_res=False):
		# Reshape input data with mask

		J = self.mask(u, m)
		cycles = J.shape[0]

		# Add extra layer to match indexes with M_x
		J = np.vstack((np.zeros((1, self.N)), J))
		J = J.flatten(order='C')
		J = J.reshape((1, (1 + cycles) * self.N), order='F')
		M_x = np.zeros(J.shape)
		X_lag = np.zeros(J.shape)

		# Select activation function
		a = self.activationFunction(act)

		# Iteratively solve differential equation with Euler's Method
		for i in range(1, (cycles + 1) * self.N * t // 1):
			vn = M_x[0, i - 1] - self.fudge * M_x[0, i - 1] * self.theta / t

			if act == "wright":  # In the case that we want wright, take out the x(t) term by redefining vn
				vn = M_x[0, i - 1]

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
	               cv=True, beta=1.0, t=1, theta=0.2, power=1):
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
		x1 = r1.calculate(u[:train_length], m, t, 'mg')[0]
		x2 = r1.calculate(u[:train_length], m, t, 'hayes')[0]
		# Is this correct? It looks like x_test and x_test_bot are defined as the same thing
		x_test1 = r1.calculate(u[train_length:], m, t, 'mg')[0]
		x_test2 = r1.calculate(u[train_length:], m, t, 'hayes')[0]

		# Train using Ridge Regression with hyperparameter tuning
		if cv:
			alphas = np.logspace(-100, 1, 100)
			NRMSE1, y_test1, y_input1, clf1 = cross_validate(alphas=alphas, x=x1, x_test=x_test1, target=target)
			# res1 = np.linalg.norm(target[50+train_length:] - y_test1[50:]) ** 2
			# ssr = np.sum((target[50+train_length:] - y_test1[50:]) ** 2)
			#
			# #  total sum of squares
			# sst = np.sum((target[50+train_length:] - np.mean(target[50+train_length:])) ** 2)
			#
			# r2_1 = 1 - (ssr / sst)
			# NRMSE2, y_test2, y_input2 = cross_validate(alphas=np.logspace(-50, 5, 500), x=x2, x_test=x_test2,
			# 											target=target)
			# res2 = np.linalg.norm(target[50+train_length:] - y_test2[50:]) ** 2
			# ssr = np.sum((target[50+train_length:] - y_test2[50:]) ** 2)
			#
			# r2_2 = 1 - (ssr / sst)
			res1 = np.linalg.norm(target[50:train_length] - y_input1[50:]) ** 2
			ssr = np.sum((target[50:train_length] - y_input1[50:]) ** 2)

			#  total sum of squares
			sst = np.sum((target[50:train_length] - np.mean(target[50:train_length])) ** 2)

			r2_1 = 1 - (ssr / sst)
			NRMSE2, y_test2, y_input2, clf2 = cross_validate(alphas=alphas, x=x2, x_test=x_test2, target=target)
			res2 = np.linalg.norm(target[50:train_length] - y_input2[50:]) ** 2
			ssr = np.sum((target[50:train_length:] - y_input2[50:]) ** 2)

			r2_2 = 1 - (ssr / sst)
		else:
			clf1 = Ridge(alpha=0)
			# clf1 = LinearRegression(n_jobs=-1)
			clf1.fit(x1, target[:train_length])
			y_test1 = clf1.predict(x1)

			# Calculate NRMSE of prediction data
			NRMSE1 = np.sqrt(
				np.mean(np.square(y_test1[50:] - target[50:train_length])) / np.var(target[50:train_length]))
			res1 = np.linalg.norm(target[50:train_length] - y_test1[50:]) ** 2
			r2_1 = 0

			clf2 = Ridge(alpha=0)
			# clf2 = LinearRegression(n_jobs=-1)
			clf2.fit(x2, target[:train_length])
			y_test2 = clf2.predict(x2)

			# Calculate NRMSE of prediction data
			NRMSE2 = np.sqrt(
				np.mean(np.square(y_test2[50:] - target[50:train_length])) / np.var(target[50:train_length]))
			res2 = np.linalg.norm(target[50:train_length] - y_test2[50:]) ** 2
			r2_2 = 0

		return NRMSE1, NRMSE2, x1, x2, target, x_test1, x_test2, y_test1, y_test2, res1, res2, r2_1, r2_2, clf1, clf2


def Values_Test(test_length=800, train_length=800, N=400, eta=0.4, tau=400,
                preload=True, mask=0.1, activate='mg', beta=1.0, t=1, theta=0.2,
                power=1, fudge=1.0, gamma=0.1):
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
	u, m, _ = load_NARMA(preload, train_length, test_length, mask, N)

	# Instantiate Reservoir, feed in training and predictiondatasets
	if type(activate) != list:
		r1 = ModifiedDelayRC(N=N, eta=eta, gamma=gamma, theta=theta,
		                     beta=beta, tau=tau, power=power, fudge=fudge)
		M_x, J, X_lag = r1.calculate(u[:train_length], m, t, activate)
	else:
		M_x, J, X_lag = [], [], []
		r1 = ModifiedDelayRC(N=N, eta=eta, gamma=gamma, theta=theta,
		                     beta=beta, tau=tau, power=power, fudge=fudge)
		for activation in activate:
			a, b, c = r1.calculate(u[:train_length], m, t, activation)
			M_x.append(a)
			J.append(b)
			X_lag.append(c)
	# plt.figure(2)
	# plt.scatter(M_x.flatten(), J.flatten(), s=.01)
	return M_x, J, X_lag


# M_x1, J1, _ = Values_Test(fudge=1.15, eta=0.5, activate='mg')
# M_x2, J2, _ = Values_Test(fudge=1.0, eta=0.5, activate='hayes')

# # plt.figure(2)
# plt.plot(M_x2.flatten() - M_x1.flatten())
# plt.show()
# if __name__ == '__main__':
# 	MRC = ModifiedDelayRC()
# 	NRMSE1, NRMSE2, x1, x2, target, x_test1, x_test2, y_test1, y_test2, res1, res2, r2_1, r2_2, clf1, clf2 \
# 	= MRC.NARMA_Test()
#
# 	fs = 10
# 	f, Cxy = signal.coherence(x1[:,64], x1[:,212], fs, nperseg=100)
# 	plt.plot(f, Cxy)
# 	plt.xlabel('frequency [Hz]')
# 	plt.ylabel('Coherence')
# 	plt.show()
# 	# results1, results2 = MRC.NARMA_Test()
#
# plt.plot(x1)
# plt.xlabel('Time step')
# plt.ylabel('Node response')
# plt.title('Responses of Nodes for MG')
# plt.show()
#
# plt.plot(M_x2)
# plt.xlabel('Time step')
# plt.ylabel('Node response')
# plt.title('Responses of Nodes for Hayes')
# plt.show()
#
# plt.plot(np.subtract(M_x2, M_x1))
# plt.xlabel('Time step')
# plt.ylabel('Node response difference')
# plt.title('Difference in Responses of Nodes for Hayes - MG')
# plt.show()
