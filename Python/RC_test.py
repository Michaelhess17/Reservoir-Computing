#!/usr/bin/env python3

__original_author__ = "Philip Jacobson"
__email__ = "philip_jacobson@berkeley.edu"

__editors__ = 'Michael Lee and Michael Hess'
__MH_email__ = "mhess21@cmc.edu"
__ML_email__ = "mlee22@cmc.edu"

import random

import numpy as np
from Delay_Reservoir import DelayReservoir
from helper_files import cross_validate, make_training_testing_set, load_NARMA, \
	plot_func, write_func
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
from tqdm import tqdm


############################################################################


def NARMA_Test(test_length=800, train_length=800,
			   plot=True, N=400, eta=0.4, gamma=0.05, tau=400, fudge=1,
			   bits=np.inf, preload=False, write=False, mask=0.1, activate='mg',
			   cv=False, beta=1.0, t=1, theta=0.2, no_act_res = False):
	"""
	Args:
		test_length: length of testing data
		train_length: length of training data
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
		t: timestep used to solve diffeq,
		theta: distance between virtual nodes in time

	Returns:
		NRMSE: Normalized Root Mean Square Error
	"""

	# Import u, m, and target
	u, m, target = load_NARMA(preload, train_length, test_length, mask, N)

	# Instantiate Reservoir, feed in training and predictiondatasets
	r1 = DelayReservoir(N=N, eta=eta, gamma=gamma, theta=theta,beta=beta, tau=tau, fudge=fudge)
	x = r1.calculate(u[:train_length], m, bits, t, activate, no_act_res = no_act_res)#[0]
	# Is this correct? It looks like x_test and x_test_bot are defined as the same thing
	x_test = r1.calculate(u[train_length:], m, bits, t, activate, no_act_res = no_act_res)#[0]                # Changed from [1] to [0]

	if no_act_res == True:
		x_test_bot = r1.calculate(u[train_length:], m, bits, t, activate, no_act_res = no_act_res)[1]
	# if fudge / eta >= 1 and act == "hayes":
	#     NRMSE = 1
	#     return NRMSE
	

	# Train using Ridge Regression with hyperparameter tuning
	if cv:
		NRMSE, y_test, y_input = cross_validate(alphas=np.logspace(-20, 5, 16), x=x, x_test=x_test, target=target)
	else:
		clf = Ridge(alpha=0)
		clf.fit(x, target[:train_length])
		y_test = clf.predict(x_test)

		# Calculate NRMSE of prediction data
		NRMSE = np.sqrt(
			np.mean(np.square(y_test[50:] - target[train_length + 50:])) / np.var(target[train_length + 50:]))

	# Write to File
	if write:
		write_func(x, x_test)
	
	if not no_act_res:
		x_test_bot = 0			# If I don't want to find the x(t)-x(t-tau) term, set flag before plotting

	# Plot predicted Time Series
	if plot:
			plot_func(x, x_test_bot, u, y_test, target, NRMSE, train_length, N)

	return NRMSE, x_test_bot


def Classification_Test(N=400, eta=0.35, gamma=0.05, tau=400, bits=np.inf, num_waves=1000, test_size=0.1,
						preload=False, write=False, mask=0.1, activate='mg', beta=1.0, power=7, t=1, theta=0.2):
	"""
	Args:
		N: number of virtual nodes
		gamma: input gain
		eta: oscillation strength
		tau: loop delay length
		bits: bit precision
		preload: preload time-series data
		write: save created time-series data
		mask: amplitude of mask values
		activate: activation function to be used (sin**2,tanh,mg)
		beta: driver gain
		t: timestep used to solve diffeq
		power: exponent for MG equation
		theta: distance between virtual nodes


	Returns:
		Accuracy of Ridge Model on Testing Data
	"""
	X_train, X_test, y_train, y_test = make_training_testing_set(num_waves=num_waves, test_percent=test_size,
																 preload=preload, write=write)

	clf = RidgeClassifier(alpha=0)
	m = np.array([random.choice([-mask, mask]) for i in range(N)])

	# Instantiate Reservoir, feed in training and predictiondatasets
	r1 = DelayReservoir(N=N, eta=eta, gamma=gamma, theta=theta, beta=beta, tau=tau, power=power)
	Xs = [X_train, X_test]
	new_Xs = [[], []]
	for k, data in enumerate(Xs):
		for idx in tqdm(range(len(data))):
			new_Xs[k].append(np.array(r1.calculate(data[idx], m, bits, t, activate)).flatten())
	new_Xs = np.array(new_Xs)
	clf.fit(new_Xs[0], y_train)

	return clf.score(new_Xs[1], y_test)


def run_test(eta, max_Tau, gamma, theta=0.2, activation="hayes", type="R"):
	"""
	Run Narma tests. Simplifies parameter input for ease of reading

	Args:
		activation : "wright", "mg", "hayes" (in the future)
		eta : term that multiplies the delayed portion
		maxTau : the maximum tau for some given k (found through matlab program)
		theta : time spacing between nodes
		type: "R" tests the model with regression; "C" tests the model with classification

	Returns:
		output: NRMSE of Narma task / Accuracy of Classification Task
	"""
	if type == "R":
		Nodes = int(
			max_Tau // theta)  # First pass set theta to 0.2, may have to experimentally find ideal theta for wright eq.
		plot_yn = True  # Plot or no plot, that is the question
		# T = 0.08            # Time normalization constant?

		output = NARMA_Test(
			test_length=800,
			train_length=800,
			gamma=gamma,  # Input gain
			plot=plot_yn,
			N=Nodes,
			eta=eta,  # parameter in front of delayed term (for wright: k)
			bits=np.inf,
			preload=False,
			cv=True,
			fudge=1,
			beta=1,
			tau=Nodes,
			activate=activation,
			theta=theta
		)
	elif type == "C":
		output = Classification_Test(
			N=max_Tau,
			eta=eta,
			gamma=gamma,
			tau=max_Tau,
			bits=np.inf,
			num_waves=333,
			test_size=0.1,
			theta=theta,
			preload=False,
			write=False,
			mask=0.1,
			activate=activation,
			beta=1.0,
			power=7,
			t=1)
	else:
		raise("This is not a valid experiment type")
	return output


#### run_test tests ####
# run_test(eta = 0.5, max_Tau = 9.993, gamma = 0.7, theta = 0.2, activation = "wright")
# run_test(eta = 0.05, max_Tau = 150, gamma = 0.5, theta = 0.2, activation = "hayes")


#### General Test for NARMA_Test ####
# NARMA_Test(
#     test_length = 800,
#     train_length = 3200,
#     gamma = 0.48707341674880045,
#     plot = True,
#     N = 317,
#     eta = 0.9615229252495553,
#     bits = np.inf,
#     preload = False,
#     cv = True,
#     beta = 0.408835328209339, 
#     tau = 317,
#     activate = 'hayes',
#     theta = 0.2
#     )

# NARMA_Test(
# 	test_length = 800,
# 	train_length = 800,
# 	gamma = 0.28,
# 	plot = True,
# 	N = 509,
# 	eta = 0.94,
# 	bits = np.inf,
# 	preload = False,
# 	cv = True,
# 	beta = 0.74, 
# 	tau = 400,
# 	activate = 'mg',
# 	theta = 0.83,
# 	no_act_res = False
# 	)

# print(run_test(eta=0.35, max_Tau=400, gamma=0.05, activation='mg', type="C"))
