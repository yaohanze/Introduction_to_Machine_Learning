#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio


# There is numpy.linalg.lstsq, which you should use outside of this classs
def lstsq(A, b):
    return np.linalg.solve(A.T @ A, A.T @ b)

#def standardinverse(A, b):
    #return (np.linalg.inv(A.T @ A)) @ (A.T @ b)

def features(x, d):
    n = x.shape[0]
    a1 = np.ones((n, 1))
    if d == 0:
        return a1
    for elem in range(1, d+1):
        a = np.power(x, elem)
        a1 = np.append(a1, a, axis=1)
    return a1

def mse(x, y, w):
    error = x @ w - y
    return np.mean(np.power(error, 2))

def main():
    data = spio.loadmat('1D_poly.mat', squeeze_me=True)
    x_train = np.array(data['x_train'])
    y_train = np.array(data['y_train']).T
    n_train = x_train.shape[0]
    x_train = x_train.reshape(n_train, 1)
    y_train = y_train.reshape(n_train, 1)


    n = 20  # max degree
    err = []
    degrees = list(range(n))
    for degree in degrees:
        x_train_features = features(x_train, degree)
        w = lstsq(x_train_features, y_train)
        #w = standardinverse(x_train_features, y_train)
        error = mse(x_train_features, y_train, w)
        err += [error]


    plt.plot(err)
    plt.xlabel('Degree of Polynomial')
    plt.ylabel('Training Error')
    plt.xlim(0, 19)
    plt.xticks(np.arange(0, 20, 1))
    plt.show()


if __name__ == "__main__":
    main()
