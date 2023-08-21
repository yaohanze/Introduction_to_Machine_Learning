import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn.utils import shuffle

data = spio.loadmat('polynomial_regression_samples.mat', squeeze_me=True)
data_x = data['x']
data_y = data['y']
data_x, datay = shuffle(data_x, data_y, random_state=7)
n = data_x.shape[0]
Kc = 4  # 4-fold cross validation
KD = 6  # max D = 6
LAMBDA = [0, 0.05, 0.1, 0.15, 0.2]

def ridge(A, b, lambda_):
	return np.linalg.solve(A.T @ A + lambda_*np.eye(A.shape[1]), A.T @ b)

def mse(x, y, w):
	error = x @ w - y
	return np.mean(error**2)

def fit(D, lambda_):
	train_error = []
	validation_error = []
	for i in range(Kc):
		p = i*n // Kc
		q = (i+1)*n // Kc
		train_x = np.delete(data_x, np.s_[p:q], 0)
		train_y = np.delete(data_y, np.s_[p:q], 0)
		valid_x = data_x[p:q]
		valid_y = data_y[p:q]
		poly = pf(D)
		train_features = poly.fit_transform(train_x)
		valid_features = poly.fit_transform(valid_x)
		w = ridge(train_features, train_y, lambda_)
		train_error += [mse(train_features, train_y, w)]
		validation_error += [mse(valid_features, valid_y, w)]
	train_error = np.array(train_error)
	validation_error = np.array(validation_error)
	return np.mean(train_error), np.mean(validation_error)

def main():
	np.set_printoptions(precision=11)
	Etrain = np.zeros((KD, len(LAMBDA)))
	Evalid = np.zeros((KD, len(LAMBDA)))
	for D in range(KD):
		print(D)
		for i in range(len(LAMBDA)):
			Etrain[D, i], Evalid[D, i] = fit(D + 1, LAMBDA[i])

	print('Average train error:', Etrain, sep='\n')
	print('Average valid error:', Evalid, sep='\n')

	result = np.where(Evalid == np.amin(Evalid))
	index = list(zip(result[0], result[1]))
	best = index[0]
	best_D = best[0]+1
	best_i = best[1]

	print('Best degree:', best_D)
	print('Best lambda:', LAMBDA[best_i])

if __name__ == "__main__":
	main()