
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

from Modified_Delay_Reservoir import mod_Delay_Res, hayes_special_Delay_Res

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

	if activate == "Hayes" and (k1 / eta) < 1:
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

def Identical_NARMA_Comp(test_length=800, train_length=800,
			   plot=True, N=400, eta=0.4, gamma=0.05, tau=400, 
			   bits=np.inf, preload=False, write=False, mask=0.1, activate='mg',
			   cv=True, beta=1.0, t=1, k1 = 1.15, theta=0.2, no_act_res = False):
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
	### Redefine redefine Parameters ###
	gamma = 0.5
	eta = 0.941
	beta = 0.83435
	N = 509
	tau = 509
	theta = 0.20034

	# Import u, m, and target
	u, m, target = load_NARMA(preload, train_length, test_length, mask, N)

	# hayes_sol = np.array(1,u.flatten().size())
	# hayes_sol = np.array()
	# mg_sol_0 = np.array()
	activate_ls = ["mg","hayes"]
	for count, activate in enumerate(activate_ls):

		if activate == "Hayes" and (k1 / eta) < 1:
			NRSME = 1
			x_test_bot = 0
			return NRSME, x_test_bot			# Should the parameters be those that put Hayes in unstable territory, 
				
		# Instantiate Reservoir, feed in training and predictiondatasets
		if activate == "mg":
			r1 = mod_Delay_Res(N=N, k1 = k1, eta=eta, gamma=gamma, theta=theta,beta=beta, tau=tau)
		if activate == "hayes":
			r1 = hayes_special_Delay_Res(N=N, k1 = k1, eta=eta, gamma=gamma, theta=theta,beta=beta, tau=tau)
		x = r1.calculate(u[:train_length], m, bits, t, activate, no_act_res = no_act_res)
		# Is this correct? It looks like x_test and x_test_bot are defined as the same thing
		x_test = r1.calculate(u[train_length:], m, bits, t, activate, no_act_res = no_act_res)                # Don't reference [0] unless no_act_res == True

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

		# Store the NRMSE predictions of mg and hayes
		if activate == "mg" and count == 0:
			mg_x = x
			mg_test = x_test
			pre_ridge_mg_0 = x.flatten()
			np.append(pre_ridge_mg_0,x_test.flatten())			# Add on the end the testing pre-ridge data to see if that lines up also
			mg_0_NRMSE = NRMSE
			mg_sol_0 = y_test.flatten()

		if activate == "hayes":
			hayes_x = x
			hayes_test = x_test
			pre_ridge_hayes = x.flatten()
			np.append(pre_ridge_hayes,x_test.flatten())
			hayes_NRMSE = NRMSE
			hayes_sol = y_test.flatten()

		if activate == "mg" and count == 1:			# This one checks out, regardless of order both mg runs score same NRMSE
			pre_ridge_mg_1 = x.flatten()
			NRMSE_sol_mg_1 = y_test.flatten()

	if not no_act_res:
		x_test_bot = 0			# If I don't want to find the x(t)-x(t-tau) term, set flag before plotting

	## Let's try to add/subtract the differences between mg (which works) and hayes (which doesn't) and run it back through. Is it just this small difference that's causing the difference in performance?
	if 'hayes' in activate_ls:
		activate = "hayes"
		x_diff = mg_x - hayes_x
		x_test_diff = mg_test - hayes_test
		# Add the differences to hayes original output
		x = hayes_x + x_diff
		x_test = hayes_test + x_test_diff

		# Calculate and train
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
		
		# Save the resulting NRMSE Values and also the y-solutions of the altered hayes
		altered_hayes_NRMSE = NRMSE
		altered_hayes_sol = y_test.flatten()

	# Plot predicted Time Series
	if plot:
		if 'hayes' in activate_ls:
			plt.figure(1)
			plt.plot(target.flatten()[train_length:], label = "NRMSE Input Sequence")
			plt.plot(mg_sol_0,label="mackey Glass")
			plt.plot(hayes_sol, label = "hayes")
			plt.title("Post-Ridge Regression mg vs. hayes: NRMSE_h = "+ str(round(hayes_NRMSE,3)) + ", NRMSE_mg = " + str(round(mg_0_NRMSE,3)))
			plt.legend()

			plt.figure(2)
			plt.plot(pre_ridge_mg_0,label="mackey Glass")
			plt.plot(pre_ridge_hayes, label = "hayes")
			plt.title("Pre-Ridge Regression mg vs. hayes")
			plt.legend()

			plt.figure(3)
			plt.plot(altered_hayes_sol, label = 'altered hayes')
			plt.plot(mg_sol_0, label = 'mackey glass')
			plt.title("corrected hayes vs mackey glass: NRMSE_altH = " + str(altered_hayes_NRMSE) + ", NRMSE_mg = " + str(mg_0_NRMSE))
			plt.legend()

			plt.figure(4)
			plt.plot(np.append(x_diff.flatten(),x_test_diff.flatten()), label = 'mg - hayes')
			plt.title("difference between mg and hayes before ridge")
			plt.legend()

			plt.show()
		else:
			plt.figure(1)
			plt.plot(target.flatten()[train_length:], label = "NRMSE Input Sequence")
			plt.plot(mg_sol_0,label="mackey glass 1st pass")
			plt.plot(NRMSE_sol_mg_1, label = "mackey glass 2nd pass")
			plt.title("Post-Ridge Regression mg vs. mg")
			plt.legend()

			plt.figure(2)
			plt.plot(pre_ridge_mg_0,label="mackey glass 1st pass")
			plt.plot(pre_ridge_mg_1, label = "mackey glass 2nd pass")
			plt.title("Pre-Ridge Regression mg vs. mg")
			plt.legend()

			plt.show()

def optimal_NARMA_Comp(test_length=800, train_length=800,
			   plot=True, N=400, eta=0.4, gamma=0.05, tau=400, 
			   bits=np.inf, preload=True, write=False, mask=0.1, activate='mg',
			   cv=False, beta=1.0, t=1, k1 = 1.15, theta=0.2, no_act_res = False):
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


	# hayes_sol = np.array(1,u.flatten().size())
	# hayes_sol = np.array()
	# NRMSE_sol_mg = np.array()
	activate = ["mg","hayes"]
	for activate in activate:
		if activate == "Hayes" and (k1 / eta) < 1:
			NRSME = 1
			x_test_bot = 0
			return NRSME, x_test_bot			# Should the parameters be those that put Hayes in unstable territory, 
				
		if activate == "hayes":
			N = 400
			tau = 400
			eta = 0.401
			gamma = 0.531
			theta = 0.2
			beta = 0.7

		if activate == "mg":
			N = 509
			tau = 509
			eta = 0.94
			gamma = 0.28
			theta = 0.834
			beta = 0.74

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

		# Store the NRMSE predictions of mg and hayes
		if activate == "mg":
			pre_ridge_mg = x.flatten()
			NRMSE_sol_mg = y_test.flatten()
		if activate == "hayes":
			pre_ridge_hayes = x.flatten()
			hayes_sol = y_test.flatten()

	if not no_act_res:
		x_test_bot = 0			# If I don't want to find the x(t)-x(t-tau) term, set flag before plotting

	# Plot predicted Time Series
	if plot:
		plt.figure(1)
		plt.plot(target.flatten()[train_length:], label = "NRMSE Input Sequence")
		plt.plot(NRMSE_sol_mg,label="mackey Glass")
		plt.plot(hayes_sol, label = "hayes")
		plt.title("Post-Ridge Regression mg vs. hayes")
		plt.legend()

		plt.figure(2)
		plt.plot(pre_ridge_mg,label="mackey Glass")
		plt.plot(pre_ridge_hayes, label = "hayes")
		plt.title("Pre-Ridge Regression mg vs. hayes")
		plt.legend()

		plt.show()

##### Tests #####

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

# NRMSE2, bot2, u2 = mod_NARMA_Test(
# 			test_length=800, 
# 			train_length=800,
# 			plot=True, 
# 			N=400, 
# 			eta=0.75, 
# 			gamma=0.05,
# 			tau=400, 
# 			k1= 1,

# 			bits=np.inf, 
# 			preload=True, 
# 			write=False, 
# 			mask=0.1, 
# 			activate='mg',
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
			eta=1, 
			gamma= 0.05,
			tau=400, 
			k1= 1.15,			# 8.5468

			bits=np.inf, 
			preload=True, 
			write=False, 
			mask=0.1, 
			activate='hayes',
			cv=True, 
			beta=1, 
			t=1, 
			theta=0.2,			#0.930
			no_act_res = False)


### Compare hayes and Mg different performance
# Identical_NARMA_Comp()

# print('hello')
#eta = 1 and k1 = 1.15, gamma = 9.95, N =400, beta = 1, theta = 0.2, NRMSE ~ 0.35 - 0.40