import os
import sys
from numpy import *
from numpy import random
from scipy import optimize as op
from scipy import io
from pylab import *
from tempfile import TemporaryFile


def standardize(X):
    mn = mean(X, axis=0)
    stdev = std(X, axis=0)
    [m, n] = shape(X)

    for i in range(0, n):
        X[:, i] = (X[:, i] - mn[i]) / (1 + stdev[i])

    return X


def sigmoid(z):
    '''computes the sigmoid function'''
    g = zeros(z.shape)
    a = 1 + e ** (-1 * z)
    g = 1 / a
    return g


def costFunction(theta, X, y, Lambda):
    '''COSTFUNCTION Compute cost and gradient for logistic regression
    J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    parameter for logistic regression and the gradient of the cost
    w.r.t. to the parameters.'''
    m, n = X.shape
    J = 1000
    grad = zeros(theta.shape)

    #Note: grad should have the same dimensions as theta
    h = sigmoid(X.dot(theta))
    x1 = X
    y1 = h - y
    grad = mat(y1) * mat(x1)
    grad = grad / m
    grad = grad[0]

    J = -1 * transpose(y).dot(log(h)) - transpose(1 - y).dot(log(1 - h))
    J = J / m
    theta = np.array(theta)
    J = J + (Lambda/(2*m)) * sum(theta * theta)

    return J


def oneVsAll(X, y, numLabels, Lambda):
    [m, n] = shape(X)
    allTheta = zeros((numLabels, n + 1))
    [rows, cols] = shape(allTheta)
    X = hstack((ones((m, 1)), X))
    initTheta = randn(n + 1, 1)

    theta = op.fmin_cg(lambda t: costFunction(t, X, (y == 10).astype(int), Lambda), initTheta,  maxiter=100, gtol=0.001, epsilon=.0001)
    theta = np.array(theta)
    allTheta[0, :] = theta

    for i in range(1, numLabels):
        print(i)
        theta = op.fmin_cg(lambda t: costFunction(t, X, (y == (i)).astype(int), Lambda), initTheta,  maxiter=100, gtol=0.001, epsilon=.0001)
        theta = np.array(theta)
        allTheta[i, :] = theta

    return allTheta


def predictOneVsAll(allTheta, X):
    [m, n] = shape(X)
    [numLabels, col_allTheta] = shape(allTheta)
    p = zeros((m, 1))
    X = hstack((ones((m, 1)), X))

    h = sigmoid(X.dot(allTheta.transpose()))
    h = h.transpose()
    p = argmax(h, axis=0)
    return (p)


data = io.loadmat('ex3data1.mat')
X = data['X']
[m, n] = shape(X)
X = X[950:1050,:]
Y = data['y']
Y = Y.flatten()
Y = reshape(Y, (len(Y), -1))
Y = Y[950:1050,:]

x1 = hstack((X, Y))
shuffle(x1)
X = x1[:, 0:n-1]
Y = x1[:, -1]

numrows = len(X)
numrow_train = int(numrows * 0.7)

X_train = X[0:numrow_train, :]
Y_train = Y[0:numrow_train]
Y_train = Y_train.flatten()

X_test = X[numrow_train:numrows, :]
Y_test = Y[numrow_train:numrows]
Y_test = Y_test.flatten()

numLabels = 10
Lambda = .3
allTheta = oneVsAll(X, Y, numLabels, Lambda)

pred = predictOneVsAll(allTheta, X_test)
correct = (pred == (Y_test % 10)).astype(double)
acc = mean(correct) * 100
print(acc)
