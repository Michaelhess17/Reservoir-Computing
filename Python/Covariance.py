import matplotlib.pyplot as plt
import numpy as np


# <h_i(t)*(x(t+\tau)>-<x(t)>*<h_i(t)>


def cov(x, y):
	xbar, ybar = x.mean(), y.mean()
	return np.sum((x - xbar)*(y - ybar))/(len(x) - 1)


# Covariance matrix
def cov_mat(X, J):
	output = np.zeros(X.shape[1])
	for i in range(X.shape[1]):
		output[i] = (cov(X[:,i], J[:,i]))
	return output


def plot_values(a, b, c, d):
	plt.figure(1)
	plt.scatter(range(len(a)), a, c='b', label='MG')
	plt.scatter(range(len(b)), b, c='g', label='Hayes')
	plt.title("Covariance between output and future input")
	plt.xlabel('Node')
	plt.ylabel('Covariance')
	mini = np.min([np.min(a), np.min(b)]) * 1.3
	maxi = np.max([np.max(a), np.max(b)]) * 1.3
	plt.ylim([mini, maxi])
	plt.legend()
	plt.show()

	plt.figure(2)
	plt.scatter(range(len(c)), c, c='b', label='MG')
	plt.scatter(range(len(d)), d, c='g', label='Hayes')
	plt.title("Covariance between input and future output")
	plt.xlabel('Node')
	plt.ylabel('Covariance')
	mini = np.min([np.min(c), np.min(d)]) * 1.3
	maxi = np.max([np.max(c), np.max(d)]) * 1.3
	plt.ylim([mini, maxi])
	plt.legend()
	plt.show()

	plt.figure(3)
	plt.scatter(range(len(a)), b - a, c='b', label='Cov(output, future input)')
	plt.scatter(range(len(c)), d - c, c='g', label='Cov(input, future output)')
	plt.title("Difference in Covariance (Hayes-MG)")
	plt.xlabel('Node')
	plt.ylabel('Covariance Difference')
	mini = np.min([np.min(b - a), np.min(d - c)]) * 1.3
	maxi = np.max([np.max(b - a), np.max(d - c)]) * 1.3
	plt.ylim([mini, maxi])
	plt.legend()
	plt.show()


# Calculate covariance matrix
# cov_mat(X, J) # (or with np.cov(X.T))

# M_x1, J1, _ = Values_Test(fudge=1.0, eta=0.5, activate='mg')
# M_x2, J2, _ = Values_Test(fudge=1.0, eta=0.5, activate='hayes')
#
# tau = 1

# Covariance of current reservoir output and future input
# a = cov_mat(M_x1[:len(M_x1)-tau, :], J1[tau+1:, :])
# b = cov_mat(M_x2[:len(M_x2)-tau, :], J2[tau+1:, :])
#
# # Covariance of current input and future reservoir output
# c = cov_mat(J1[:len(J1)-tau-1, :], M_x1[tau:, :])
# d = cov_mat(J2[:len(J2)-tau-1, :], M_x2[tau:, :])

# plot_values(a, b, c, d)

# var_cov_mat = np.zeros([M_x1.shape[1], M_x1.shape[1], 2])
# for num, data in enumerate([M_x1, M_x2]):
# 	for x_idx in range(data.shape[1]):
# 		for y_idx in range(data.shape[1]):
# 			var_cov_mat[x_idx, y_idx, num] = cov(data[:, x_idx], data[:, y_idx])

