import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
from sklearn.preprocessing import PolynomialFeatures as pf

data = spio.loadmat('polynomial_regression_samples.mat', squeeze_me=True)
data_x = data['x']
data_y = data['y']
#data_x, datay = shuffle(data_x, data_y, random_state=3)
n = data_x.shape[0]
Kc = 4  # 4-fold cross validation
KD = 6  # max D = 6
LAMBDA = [0, 0.05, 0.1, 0.15, 0.2]
# train set validation set split
i = n // Kc
data_y = data_y.reshape(n, 1)
data_x1 = data_x[0:i, :]
data_x2 = data_x[i:2*i, :]
data_x3 = data_x[2*i:3*i, :]
data_x4 = data_x[3*i:4*i, :]
data_y1 = data_y[0:i, :]
data_y2 = data_y[i:2*i, :]
data_y3 = data_y[2*i:3*i, :]
data_y4 = data_y[3*i:4*i, :]
data_x_lst = [data_x1, data_x2, data_x3, data_x4]
data_y_lst = [data_y1, data_y2, data_y3, data_y4]
train_data_x = [np.concatenate((data_x_lst[1], data_x_lst[2], data_x_lst[3]), axis=0), np.concatenate((data_x_lst[0], data_x_lst[2], data_x_lst[3]), axis=0), np.concatenate((data_x_lst[0], data_x_lst[1], data_x_lst[3]), axis=0), np.concatenate((data_x_lst[0], data_x_lst[1], data_x_lst[2]), axis=0)]
train_data_y = [np.concatenate((data_y_lst[1], data_y_lst[2], data_y_lst[3]), axis=0), np.concatenate((data_y_lst[0], data_y_lst[2], data_y_lst[3]), axis=0), np.concatenate((data_y_lst[0], data_y_lst[1], data_y_lst[3]), axis=0), np.concatenate((data_y_lst[0], data_y_lst[1], data_y_lst[2]), axis=0)]
validation_data_x = data_x_lst
validation_data_y = data_y_lst

def ridge(A, b, lambda_):
    return np.linalg.solve(A.T @ A + lambda_*np.eye(A.shape[1]), A.T @ b)

def mse(x, y, w):
    error = x @ w - y
    return np.mean(error**2)

def fit(D, lambda_):
    # YOUR CODE TO COMPUTE THE AVERAGE ERROR PER SAMPLE
    train_error = []
    validation_error = []
    m = data_x.shape[0] // Kc
    for i in range(Kc):
        train_x = train_data_x[i]
        train_y = train_data_y[i]
        valid_x = validation_data_x[i]
        valid_y = validation_data_y[i]
        #train_x1 = train_x[:, 0].reshape(m*3, 1)
        #train_x2 = train_x[:, 1].reshape(m*3, 1)
        #train_x3 = train_x[:, 2].reshape(m*3, 1)
        #train_x4 = train_x[:, 3].reshape(m*3, 1)
        #train_x5 = train_x[:, 4].reshape(m*3, 1)
        #valid_x1 = valid_x[:, 0].reshape(m, 1)
        #valid_x2 = valid_x[:, 1].reshape(m, 1)
        #valid_x3 = valid_x[:, 2].reshape(m, 1)
        #valid_x4 = valid_x[:, 3].reshape(m, 1)
        #valid_x5 = valid_x[:, 4].reshape(m, 1)
        #train_features = np.empty((m*3, 0))
        #valid_features = np.empty((m, 0))
        #for a in range(D+1):
            #for b in range(D+1):
                #for c in range(D+1):
                    #for d in range(D+1):
                        #for e in range(D+1):
                            #if a+b+c+d+e <= D:
                                #train_feature1 = np.power(train_x1, a)
                                #train_feature2 = np.power(train_x2, b)
                                #train_feature3 = np.power(train_x3, c)
                                #train_feature4 = np.power(train_x4, d)
                                #train_feature5 = np.power(train_x2, e)
                                #train_feature = np.multiply(np.multiply(np.multiply(np.multiply(train_feature1, train_feature2), train_feature3), train_feature4), train_feature5)
                                #train_features = np.concatenate((train_features, train_feature), axis=1)
                                #valid_feature1 = np.power(valid_x1, a)
                                #valid_feature2 = np.power(valid_x2, b)
                                #valid_feature3 = np.power(valid_x3, c)
                                #valid_feature4 = np.power(valid_x4, d)
                                #valid_feature5 = np.power(valid_x5, e)
                                #valid_feature = np.multiply(np.multiply(np.multiply(np.multiply(valid_feature1, valid_feature2), valid_feature3), valid_feature4), valid_feature5)
                                #valid_features = np.concatenate((valid_features, valid_feature), axis=1)
        #train_x = np.delete(data_x, np.s_[i*n//4:(i+1)*n//4], 0)
        #train_y = np.delete(data_y, np.s_[i*n//4:(i+1)*n//4], 0)
        #valid_x = data_x[i*n//4:(i+1)*n//4]
        #valid_y = data_y[i*n//4:(i+1)*n//4]
        poly = pf(D)
        train_features = poly.fit_transform(train_x)
        valid_features = poly.fit_transform(valid_x)
        w_train = ridge(train_features, train_y, lambda_)
        train_error += [mse(train_features, train_y, w_train)]
        validation_error += [mse(valid_features, valid_y, w_train)]
    train_error = np.array(train_error)
    validation_error = np.array(validation_error)
    return (np.mean(train_error), np.mean(validation_error))



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

    #Etrain = np.zeros((KD, 1))
    #Evalid = np.zeros((KD, 1))
    #for D in range(KD):
        #print(D)
        #Etrain[D, 0], Evalid[D, 0] = fit(D+1, 0.05)
    #print('Average train error:', Etrain, sep='\n')
    #print('Average valid error:', Evalid, sep='\n')

if __name__ == "__main__":
    main()
