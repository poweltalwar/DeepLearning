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
from PIL import Image



def plotData(image):
    '''plots the input data '''
    # the image matrix will have to be transposed to be viewed correcty
    # cmap shows the color map
    plt.imshow(image, cmap='Greys')
    plt.show()



def arrangeData(X, Y, cutoff):
    '''divides the data into training, cross validation and test datsets '''
    
    # get input matrix shape and no. of output classes
    m, n = X.shape
    numLabels = len(unique(Y))

    # examples per output-class
    epc = int(m / numLabels)
    epc_train = int(epc * cutoff)
    epc_test = epc - epc_train - 1
    
    # choosing training and test dataset size
    numrow_train = m * cutoff
    numrow_test = m - numrow_train
    
    # initializing training and test dataset
    X_train = zeros((numrow_train, n))
    X_test = zeros((numrow_test , n))

    Y_train = zeros(numrow_train)    
    Y_test = zeros(numrow_test)

    '''
    # thresholding the dataset
    X = (X-amin(X))/(amax(X)-amin(X))
    X[X>0.5] = 1
    [X<=0.5] = 0
    '''

    # assigning examples to training dataset
    for i in range(0, numLabels):
        for j in range(0, epc_train):
            X_train[i * epc_train + j] = X[i * epc + j]
            Y_train[i * epc_train + j] = Y[i * epc + j]

    # assigning exampples to test dataset
    for i in range(0, numLabels):
        for j in range(0, epc_test):
            X_test[i * epc_test + j] = X[epc_train + i * epc + j]
            Y_test[i * epc_test + j] = Y[epc_train + i * epc + j]

    return X_train, Y_train, X_test, Y_test


def arrangeData2(X, Y, ratio):
    #shuffle data
    x1 = hstack((X, Y))
    random.shuffle(x1)
    X = x1[:, 0:-1]
    Y = x1[:, -1]
    numrows = len(X)
    numrow_train = int(numrows * ratio)

    #training set
    X_train = X[0:numrow_train, :]
    Y_train = Y[0:numrow_train]
    Y_train = Y_train.flatten()

    #test set
    X_test = X[numrow_train:numrows, :]
    Y_test = Y[numrow_train:numrows]
    Y_test = Y_test.flatten()
    
    return X_train, Y_train, X_test, Y_test



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
    #J = 1000
    #grad = zeros(theta.shape)

    #Note: grad should have the same dimensions as theta
    h = sigmoid(X.dot(theta))
    
    '''x1 = X
    y1 = h - y
    grad = mat(y1) * mat(x1)
    grad = grad / m
    grad = grad[0]'''
    
    J = -1 * transpose(y).dot(log(h)) - transpose(1 - y).dot(log(1 - h))
    J = J / m
    theta = array(theta)
    J = J + (Lambda / (2 * m)) * sum(theta * theta)
   # print(J)

    return J


def oneVsAll(X, y, numLabels, Lambda):
    '''initialise the theta parameters and train them on the training set for
    all labels'''
    [m, n] = shape(X)
    
    #append the bias term to input data
    X = hstack((ones((m, 1)), X))

    [m, n] = shape(X)
    allTheta = zeros((numLabels, n))  
    
    #initial value of theta, further to be optimized
    initTheta = random.randn(n, 1)

    '''theta = op.fmin_cg(lambda t: costFunction(t, X, (y == 0).astype(int), Lambda), initTheta,
                         maxiter=100, gtol=0.001, epsilon=.001)
    theta = array(theta)
    
    #append in main array of theta
    allTheta[0, :] = theta'''

    #optimize using fmin_cg
    for i in range(0, numLabels):
        print(i)
        theta = op.fmin_cg(lambda t: costFunction(t, X, ((y == i).astype(int)), Lambda), initTheta,
                           maxiter=1000, gtol=0.00001, epsilon=1.5e-8)
        theta = array(theta)
        allTheta[i, :] = theta

    return allTheta


def predictOneVsAll(allTheta, X):
    [m, n] = shape(X)
    X = hstack((ones((m, 1)), X))
    [m, n] = shape(X)
    [numLabels, col_allTheta] = shape(allTheta)
    p = zeros((m, 1))

    h = sigmoid(X.dot(allTheta.transpose()))
    h = h.transpose()
    h[h == 1] = 0
    p = argmax(h, axis=0)
    return (p)


# load data    
data = io.loadmat('database/num20201.mat')
size = (20, 20)
X = data['X']
[m, n] = shape(X)
Y = data['Y']
Y = reshape(Y, (len(Y), -1))

numLabels = len(unique(Y))
Y[Y == 10] = 0

#X[X < 0.5] = 0
#X[X >= .5] = 1

#divide into training and test set, division of 70:30
cutoff = 0.75
X_train, Y_train, X_test, Y_test = arrangeData(X, Y, cutoff)

#get theta parameters for multiclass logistic regression
Lambda = 10
allTheta = oneVsAll(X_train, Y_train, numLabels, Lambda)

#after training, run on test set and check accuracy
pred = predictOneVsAll(allTheta, X_test)
correct = (pred == Y_test).astype(double)
acc = mean(correct) * 100
print(acc)
save('solutions/logTrain202014.npy', allTheta)
