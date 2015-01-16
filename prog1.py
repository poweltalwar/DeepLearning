import os
import sys
import numpy
from numpy import *
from numpy import random
from scipy import optimize as op
from scipy import io
import pylab
from matplotlib import *
import matplotlib.pyplot as plt
from tempfile import TemporaryFile


def plotData(X, Y):
    '''plots the input data '''

    # plot any one case (20x20 image) from the input
    # the image matrix will have to be transposed to be viewed correcty
    # cmap shows the color map
    c = random.randint(0, X.shape[0])    
    image = array(X[c,:])
    plt.imshow((image.reshape(20,20)).T, cmap = 'Greys')
    plt.show()
    
    #plot the same ouptut case
    print("true number is \t" + str(Y[c]))
    
    return c


def standardize(X):
    '''standardize data if need be. This function brings all the attributes
    to zero mean and unit variance'''
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
    theta = array(theta)
    J = J + (Lambda/(2*m)) * sum(theta * theta)

    return J


def oneVsAll(X, y, numLabels, Lambda):
    '''initialise the theta parameters and train them on the training set for
    all labels'''
    [m, n] = shape(X)
    allTheta = zeros((numLabels, n + 1))

    #append the bias term to input data
    X = hstack((ones((m, 1)), X))
    #initial value of theta, further to be optimized
    initTheta = random.randn(n + 1, 1)

    theta = op.fmin_cg(lambda t: costFunction(t, X, (y == 10).astype(int), Lambda), initTheta,
                         maxiter=100, gtol=0.001, epsilon=.0001)
    theta = array(theta)
    
    #append in main array of theta
    allTheta[0, :] = theta

    #optimize using fmin_cg
    for i in range(1, numLabels):
        print(i)
        theta = op.fmin_cg(lambda t: costFunction(t, X, (y == (i)).astype(int), Lambda), initTheta,  maxiter=100, gtol=0.001, epsilon=.0001)
        theta = array(theta)
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


#load data here
data = io.loadmat('ex3data1.mat')
X = data['X']
[m, n] = shape(X)
Y = data['y']
Y = reshape(Y, (len(Y), -1))
Y[Y == 10] = 0 

#shuffle data
x1 = hstack((X, Y))
random.shuffle(x1)
X = x1[:, 0:-1]
Y = x1[:, -1]

X = X[1:100,:]
Y = Y[1:100]

#divide into training and test set, division of 70:30
numrows = len(X)
numrow_train = int(numrows * 0.7)

#training set
X_train = X[0:numrow_train, :]
Y_train = Y[0:numrow_train]
Y_train = Y_train.flatten()

#test set
X_test = X[numrow_train:numrows, :]
Y_test = Y[numrow_train:numrows]
Y_test = Y_test.flatten()

#get theta parameters for multiclass logistic regression
numLabels = len(unique(Y))
Lambda = .3
allTheta = oneVsAll(X, Y, numLabels, Lambda)

#after training, run on test set and check accuracy
pred = predictOneVsAll(allTheta, X_test)
correct = (pred == (Y_test % 10)).astype(double)
acc = mean(correct) * 100
print(acc)

testNum = plotData(X, Y)
X1 = X[testNum, :]
X1 = reshape(X1, (len(X1), -1))
X1 = X1.transpose()
[m, n] = shape(array(X1))
X1 = hstack((ones((m, 1)), X1))
h = sigmoid(X1.dot(allTheta.transpose()))
h = h.transpose()
print(h)
p = argmax(h, axis=0)
print(" predicted output is \t" + str(p[0]))


