import os
import sys
from numpy import *
from scipy import optimize
from pylab import *
from prog2 import *



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
    w.r.t. to the parameters.
    '''    
    m,n = X.shape
    J = 1000;
    grad = zeros(theta.shape)
    
    #Note: grad should have the same dimensions as theta
    h = sigmoid( X.dot(theta) )
   
    grad = (mat(h - y) * mat(X) / m
    grad = grad[0]

    J =  -1 * transpose( y ).dot( log(h) ) - transpose( 1-y ).dot( log(1-h) )  
    J =  J / m
    theta = np.array( theta );
    J = J + ( Lambda/(2*m) ) * sum( theta * theta )  ;

    
    return  J


def predict(theta, X):
    '''calculates the accuracy given the inout data and the learned
    weihts'''
    #print(shape(X))
    #print(shape(theta))
    p = sigmoid( X.dot(theta) ) >= 0.5
    #p = 10
    return p



#import the data
data = loadtxt('ex3data1.mat', delimiter=',')
numrows = len(data)
numrow_train = int(numrows*0.7)
numcols = len(data[0])

#separate the data
X_train = data[0:numrow_train,0:numcols-1]
Y_train = data[0:numrow_train,numcols-1]
X_pre = data[numrow_train:numrows,0:numcols-1]
Y_pre = data[numrow_train:numrows,numcols-1]

#add the bias term to data
m = int(numrows*0.7)
n = numcols-1
X0 = ones((m,1))
X0_pre = ones((numrows-m, 1))
X_train = hstack((X0,X_train))
X_pre = hstack((X0_pre,X_pre))

#initialise the fitting parametrers
initial_theta = zeros( (n + 1, 1) );

#initialise lambda for regularisation
Lambda = .3;

#compute initial cost and gradient
cost = costFunction(initial_theta, X_train, Y_train, Lambda)

options = {'full_output': True, 'maxiter': 400}


theta, cost, _, _, _ = optimize.fmin(lambda t: costFunction(t, X_train, Y_train, Lambda ), initial_theta, **options)

p = predict(theta, X_pre)
print('Train Accuracy:', (p == Y_pre).mean() * 100)
