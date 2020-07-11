
import numpy as np
import seaborn as sns
from Delay_Reservoir import DelayReservoir
from RC_test import run_test, NARMA_Test
from hyperopt import tpe, hp, fmin
from matplotlib import pyplot as plt

from helper_files import cross_validate, make_training_testing_set, load_NARMA, \
	plot_func, write_func, margin
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier

from Modified_Delay_Reservoir import mod_Delay_Res

"""
Intended to store parameter tweaks for Hayes 07/10/20 until able to merge this code with that in Parameter_Searching.py when Ray/Tune works
the tweak in particular is adding a k1 term in front of x(t). We do this by adding in a class variable before a super().__init__() call in 'modified_delay_reservoir'
and have this modified version of narma_test call our child class when it initiates the reservoir.
"""

def mod_NARMA_Test(test_length=800, train_length=800,
			   plot=True, N=400, eta=0.4, gamma=0.05, tau=400, 
			   bits=np.inf, preload=False, write=False, mask=0.1, activate='mg',
			   cv=False, beta=1.0, t=1, k1 = 1, theta=0.2, no_act_res = False):
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
	if activate == "Hayes" and k1 / eta:
		NRSME = 1
		x_test_bot = 0
		return NRSME, x_test_bot			# Should the parameters be those that put Hayes in unstable territory, 
			
	# Import u, m, and target
	u, m, target = load_NARMA(preload, train_length, test_length, mask, N)



	# Instantiate Reservoir, feed in training and predictiondatasets
	r1 = mod_Delay_Res(N=N, k1 = k1, eta=eta, gamma=gamma, theta=theta,beta=beta, tau=tau)
	x = r1.calculate(u[:train_length], m, bits, t, activate, no_act_res = no_act_res)#[0]
	# Is this correct? It looks like x_test and x_test_bot are defined as the same thing
	x_test = r1.calculate(u[train_length:], m, bits, t, activate, no_act_res = no_act_res)#[0]                # Changed from [1] to [0]

	if no_act_res == True:
		x_test_bot = r1.calculate(u[train_length:], m, bits, t, activate, no_act_res = no_act_res)[1]

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

	return NRMSE, x_test_bot, u


# NRMSE2, bot2, u2 = mod_NARMA_Test(
# 			test_length=800, 
# 			train_length=800,
# 			plot=True, 
# 			N=400, 
# 			eta=0.75, 
# 			gamma=0.05,
# 			tau=400, 
# 			k1= 1.15,			

# 			bits=np.inf, 
# 			preload=True, 
# 			write=False, 
# 			mask=0.1, 
# 			activate='hayes',
# 			cv=True, 
# 			beta=1, 
# 			t=1, 
# 			theta=0.2,
# 			no_act_res = False)

NRMSE2, bot2, u2 = mod_NARMA_Test(
			test_length=800, 
			train_length=800,
			plot=True, 
			N=400, 
			eta=0.75, 
			gamma=0.05,
			tau=400, 
			k1= 1,			

			bits=np.inf, 
			preload=True, 
			write=False, 
			mask=0.1, 
			activate='mg',
			cv=True, 
			beta=1, 
			t=1, 
			theta=0.2,
			no_act_res = False)

print('hello')
#eta = 1 and k1 = 1.15, gamma = 9.95, N =400, beta = 1, theta = 0.2, NRMSE ~ 0.35 - 0.40