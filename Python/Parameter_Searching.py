import numpy as np
import seaborn as sns
from Delay_Reservoir import DelayReservoir
from RC_test import run_test, NARMA_Test
from hyperopt import tpe, hp, fmin
from matplotlib import pyplot as plt


def wright_grid_search(max_Tau=9.992, wright_k=-0.15, eta=0.05, theta=0.2, gamma=0.05, parameter_searched="theta",
                       activation="hayes"):
	"""
	Runs a grid search manipulating theta for the Wright equation. max_tau is the longest tau possible at wright_k.
	Holds all else constant and varies theta. Next ones will have to vary gamma.

	args:
		wright_k: k term in the Wright equation
		max_Tau: longest value of tau at wright_k
		eta: holding eta fixed case, default value of eta
		parameter_searched = which parameter(s) would you like to do a grid search for? ("theta", "eta_gamma")

	output:
		if "theta":
			graph showing NRMSE on vertical vs choices of theta
		if "eta_gamma":
			heatmap showing NRMSE various combinations of k(eta in program) and gamma
	"""

	if parameter_searched == "theta":
		theta_range = np.linspace(0.2, 0.3, 10)  # Generate theta range
		wright_k = eta  # redefinition for ease of understanding that eta serves the role of k in the wright equation
		output_vals = {}  # Dictionary to hold output values
		for num, i in enumerate(theta_range):
			output = run_test(activation=activation, theta=i, eta=wright_k, max_Tau=max_Tau, gamma=gamma)
			output_vals[theta_range[num]] = output  # Collect theta value and corresponding NRMSE in a dictionary entry

		# Unzip dictionary entries
		lists = sorted(output_vals.items())
		x, y = zip(*lists)

		# Plot data
		plt.plot(x, y)
		plt.xlabel('theta')
		plt.ylabel('NRMSE')
		plt.show()

	if parameter_searched == "eta_gamma":
		interps = 10  # How many combinations of k and gamma would you like?
		k_range = np.linspace(-0.20, -0.15, interps)
		gamma_range = np.linspace(0.00, 0.05, interps)
		eg_output = np.zeros((interps, interps))

		for k_index, kr in enumerate(k_range):
			for g_index, gr in enumerate(gamma_range):
				output = run_test(activation=activation, theta=theta, eta=kr, max_Tau=max_Tau, gamma=gr)
				eg_output[k_index, g_index] = output  # eg_output fills left corner to right corner, \n and so on

		# plot data
		ax = sns.heatmap(eg_output, cmap='coolwarm', xticklabels=np.round_(gamma_range, 4),
		                 yticklabels=np.round_(k_range, 4))
		ax.set(xlabel="gamma", ylabel="k", title="Wright simulation results for NARMA task")
		plt.show()


def mg_hayes_comp():
	# Set large parameters for calculate
	t = 1
	bits = np.inf
	train_length = 800

	# Import data

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

	# MG portion, collect output
	activate = 'mg'
	r1 = DelayReservoir(N=400, eta=1, gamma=0.05, theta=0.2, \
	                    beta=1, tau=400)
	x_mg, vn_mg = r1.calculate(u[:train_length], m, bits, t, activate, no_act_res=True)  # Takes the actual output
	# x_mg_vn = r1.calculate(u[:train_length], m, bits, t, activate)[1]

	# Hayes portion, collect output
	activate = 'mod_hayes'
	x_hayes, vn_hayes = r1.calculate(u[:train_length], m, bits, t, activate, no_act_res=True)
	# x_hayes_vn = r1.calculate(u[:train_length], m, bits, t, activate)[1]

	# Flatten the values
	x_mg = x_mg.flatten()
	x_hayes = x_hayes.flatten()

	vn_mg = vn_mg.flatten()
	vn_hayes = vn_hayes.flatten()

	# Plot the data
	plt.figure(1)
	plt.plot(x_mg, label="mackey-glass")
	plt.plot(x_hayes, label="hayes")
	plt.xlabel("cycle * Nodes")
	plt.title("Raw Mg vs Hayes Ouputs with Same NARMA Input")
	plt.legend()

	plt.figure(2)
	plt.plot(vn_mg, label="mackey-glass vn")
	plt.plot(vn_hayes, label="hayes vn")
	plt.xlabel("cycle * Nodes")
	plt.title("Raw VN: Mg vs Hayes Ouputs with Same NARMA Input")
	plt.legend()

	plt.show()


def ml_test_hayes(param):
	"""
	Sets up 'function' to minimize.
	Declares variables to hyperopt.
	Returns the mean NRMSE of 3 NARMA10 tasks
	"""
	gamma, eta, beta, theta, hayes_param = \
		param['gamma'], param['eta'], param['beta'], param['theta'], param['hayes_param']

	N = 400  # Number of Nodes

	return np.mean([NARMA_Test(
		test_length=1000,
		train_length=3200,
		gamma=gamma,
		plot=False,
		N=N,
		eta=eta,
		bits=np.inf,
		preload=False,
		beta=beta,
		fudge=hayes_param,
		tau=N,
		activate='hayes',
		theta=theta
	)[0] for _ in range(3)])


def hyperopt_grad_hayes():
	"""
	calls the fmin
	"""
	best = fmin(
		fn=ml_test_hayes,
		space={
			# 'x': hp.randint('x',800),
			'gamma': hp.uniform('gamma', 0.01, 3),
			'eta': hp.uniform('eta', 0, 1),
			# 'N': hp.randint('N',800),
			'theta': hp.uniform('theta', 0, 1),
			'beta': hp.uniform('beta', 0, 1),
			'hayes_param': hp.uniform('hayes_param', -1, 0)
		},
		algo=tpe.suggest,
		max_evals=100
	)

	print(best)


def ml_test_wright(param):
	"""
	Sets up 'function' to minimize.
	Declares variables to hyperopt.
	Returns the mean NRMSE of 3 NARMA10 tasks
	"""
	gamma, k, N, theta, beta = \
		param['gamma'], param['k'], param['N'], \
		param['theta'], param['beta']

	return np.mean([NARMA_Test(
		test_length=800,
		train_length=3200,
		gamma=gamma,
		plot=False,
		N=N,
		eta=k,
		bits=np.inf,
		preload=False,
		beta=beta,
		tau=N,
		activate='hayes',
		theta=theta
	) for _ in range(3)])


def hyperopt_grad_wright():
	"""
	calls the fmin
	"""
	best = fmin(
		fn=ml_test_hayes,
		space={
			# 'x': hp.randint('x',800),
			'gamma': hp.uniform('gamma', 0.01, 2),
			'eta': hp.uniform('k', 0, 1),
			'N': hp.randint('N', 800),
			'theta': hp.uniform('theta', 0.0001, 1),
			'beta': hp.uniform('beta', 0, 1)
		},
		algo=tpe.suggest,
		max_evals=100
	)

	print(best)

##### TESTS #####

#### mg_hayes_comp tests ####
mg_hayes_comp()

#### Grid search tests ####
# wright_grid_search(parameter_searched="eta_gamma", activation = "hayes")




#### Hyperopt Tests ####
# hyperopt_grad_wright()
# hyperopt_grad_hayes()
