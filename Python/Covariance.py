import matplotlib.pyplot as plt
import numpy as np

from Store_values import Values_Test


# <h_i(t)*(x(t+\tau)>-<x(t)>*<h_i(t)>


def cov(x, y):
	xbar, ybar = x.mean(), y.mean()
	return np.sum((x - xbar) * (y - ybar)) / (len(x))


# Covariance matrix
def cov_mat(X, J):
	output = np.zeros((X.shape[1], J.shape[1]))
	for i in range(X.shape[1]):
		for j in range(J.shape[1]):
			output[i, j] = (cov(X[:, i], J[:, j]))
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

# M_x1, J1, _ = Values_Test(fudge=1.0, eta=0.5, activate='mg', preload=False)
M_x, J, _ = Values_Test(fudge=1.0, eta=0.5, activate=['hayes', 'mg'], preload=False)
M_x1 = M_x[0]
M_x2 = M_x[1]
J1 = J[0]
J2 = J[1]

tau = 0

# Covariance of current reservoir output and future input
a = cov_mat(M_x1[:len(M_x1) - tau, :], J1[tau + 1:, :])
b = cov_mat(M_x2[:len(M_x2) - tau, :], J2[tau + 1:, :])

# Covariance of current input and future reservoir output
c = cov_mat(J1[:len(J1) - tau - 1, :], M_x1[tau:, :])
d = cov_mat(J2[:len(J2) - tau - 1, :], M_x2[tau:, :])

# plot_values(a, b, c, d)

var_cov_mat = np.zeros([M_x1.shape[1], M_x1.shape[1], 2])
for num, data in enumerate([J1, M_x1]):
	for x_idx in range(data.shape[1]):
		for y_idx in range(data.shape[1]):
			var_cov_mat[x_idx, y_idx, num] = cov(data[:, x_idx], data[:, y_idx])

e = np.abs(d / c)
f = np.abs(a / b)
g = var_cov_mat[:, :, 0]
h = var_cov_mat[:, :, 1]
i = g / h

# high_nodes = np.where(e > (e.max() - e.max()*0.2))[0].tolist()
# low_nodes = np.where(e < (e.min() - e.min()*0.5))[0].tolist()

high_nodes = np.where(e > 1.2)[0].tolist()
low_nodes = np.where(e < 0.8)[0].tolist()
all_nodes = list(set(low_nodes + high_nodes))

# sb.heatmap(e.reshape(20, 20))
# plt.show()
#
# sb.heatmap(f.reshape(20, 20))
# plt.show()

print(all_nodes)

lim = 3

high_nodes = [k for k in zip(*np.where(e > lim))]
length = min(4, len(high_nodes))
high_nodes = high_nodes[:length]
low_nodes = [k for k in zip(*np.where(e < 1/2))]
fig, axs = plt.subplots(length, 2, clear=True, sharey='col', figsize=(10, 8), constrained_layout=True)
for idx, node in enumerate(low_nodes):
	axs[idx, 0].plot(M_x2[:, node[0]]-M_x1[:, node[0]])
for idx, node in enumerate(high_nodes):
	axs[idx, 1].plot(M_x2[:, node[0]]-M_x1[:, node[0]])


fig.suptitle('MG-Hayes (Left: nodes with high cov diff / Right: nodes with low cov diff)', y=1.1)
plt.show()


# new_Xs, clf, X_train, X_test, y_train, y_test, m = Sin_Test(num_waves=15, activate='mg')


# X_train_new = np.zeros((400, 100 * 13))
# for idx, wave in enumerate(new_Xs[0]):
# 	for idx2, features in enumerate(wave):
# 		X_train_new[:, idx * 100 + idx2] = features
# X_train_new = X_train_new.T
# y_train_new = np.array([np.array([out]*100) for out in y_train]).flatten()


