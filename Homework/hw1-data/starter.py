import numpy as np
import matplotlib.pyplot as plt

# Load the training dataset
train_features = np.load("train_features.npy")
train_labels = np.load("train_labels.npy").astype("int8")

n_train = train_labels.shape[0]
print('n_train is:',n_train)

def visualize_digit(features, label):
    # Digits are stored as a vector of 400 pixel values. Here we
    # reshape it to a 20x20 image so we can display it.
    plt.imshow(features.reshape(20, 20), cmap="binary")
    plt.xlabel("Digit with label " + str(label))
    plt.show()

# Visualize a digit
# visualize_digit(train_features[0,:], train_labels[0])

# Plot three images with label 0 and three images with label 1
i = 3
m = 0
while (i > 0):
	label = train_labels[m]
	if label == 0:
		visualize_digit(train_features[m,:],label)
		i -= 1
	m += 1
j = 3
n = 0
while(j > 0):
	label = train_labels[n]
	if label == 1:
		visualize_digit(train_features[n,:],label)
		j -= 1
	n += 1
# Linear regression

# Solve the linear regression problem, regressing
# X = train_features against y = 2 * train_labels - 1
X = train_features
y = 2 * train_labels - 1
w = np.linalg.solve(np.dot(X.T,X),np.dot(X.T,y))

# Report the residual error and the weight vector
residual = np.dot(X,w) - y
residualError = np.dot(residual.T,residual)
print('Residual error is:', residualError)
print('The first 20 entries of w are as follows:')
w1 = w[:20]
for elem in np.nditer(w1):
	print(elem)
# Load the test dataset
# It is good practice to do this after the training has been
# completed to make sure that no training happens on the test
# set!
test_features = np.load("test_features.npy")
test_labels = np.load("test_labels.npy").astype("int8")

n_test = test_labels.shape[0]
print('n_test is:',n_test)

# Implement the classification rule and evaluate it
# on the training and test set
def successPercent(features, labels, w, t):
	guess = np.dot(features,w)
	for i in range(len(guess)):
		if guess[i] <= t:
			guess[i] = 0
		else:
			guess[i] = 1
	for j in range(len(guess)):
		if guess[j] == labels[j]:
			guess[j] = 1
		else:
			guess[j] = 0
	return np.sum(guess) / len(guess)

percentTrain = successPercent(train_features,train_labels,w,0)
percentTest = success Percent(test_features,test_labels,w,0)
print('Training set correctly classified percentage is:', percentTrain)
print('Test set correctly classified percentage is:', percentTest)
# Try regressing against a vector with 0 for class 0
# and 1 for class 1
y1 = train_labels
w2 = np.linalg.solve(np.dot(X.T,X),np.dot(X.T,y1))
percentTrain1 = successPercent(train_features,train_labels,w2,0.5)
percentTest1 = successPercent(test_features,test_labels,w2,0.5)
print('After using 0 for class 0 and 1 for class 1:')
print('Training set correctly classified percentage is:', percentTrain1)
print('Test set correctly classified percentage is:', percentTest1)
# Form a new feature matrix with a column of ones added
# and do both regressions with that matrix
biasTrain = np.ones((n_train,1))
biasTest = np.ones((n_test,1))
biasTrainFeatures = np.append(train_features,biasTrain,axis=1)
biasTestFeatures = np.append(test_features,biasTest,axis=1)
X1 = biasTrainFeatures
w3 = np.linalg.solve(np.dot(X1.T,X1),np.dot(X1.T,y))
w4 = np.linalg.solve(np.dot(X1.T,X1),np.dot(X1.T,y1))
biasPercentTrain = successPercent(biasTrainFeatures,train_labels,w3,0)
biasPercentTest = successPercent(biasTestFeatures,test_labels,w3,0)
biasPercentTrain1 = successPercent(biasTrainFeatures,train_labels,w4,0.5)
biasPercentTest1 = successPercent(biasTestFeatures,test_labels,w4,0.5)
print('After adding the bias column to the feature matrix:')
print('With -1/1 target:')
print('Training set correctly classified percentage is:', biasPercentTrain)
print('Test set correctly classified percentage is:', biasPercentTest)
print('With 0/1 target:')
print('Training set correctly classified percentage is:', biasPercentTrain1)
print('Test set correctly classified percentage is:', biasPercentTest1)
# Logistic Regression

# You can also compare against how well logistic regression is doing.
# We will learn more about logistic regression later in the course.

import sklearn.linear_model

lr = sklearn.linear_model.LogisticRegression()
lr.fit(X, train_labels)

test_error_lr = 1.0 * sum(lr.predict(test_features) != test_labels) / n_test
print('Test set correctly classified percentage with logistic regression is:', 1-test_error_lr)