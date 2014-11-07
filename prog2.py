import numpy as np
from numpy import *


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
   
    x1 = X
    y1 = h-y
    grad = mat(y1) * mat(x1)
    grad = grad/m
    grad = grad[0]

    J =  -1 * transpose( y ).dot( log(h) ) - transpose( 1-y ).dot( log(1-h) )  
    J =  J / m
    theta = np.array( theta );
    J = J + ( Lambda/(2*m) ) * sum( theta * theta )  ;

    
    return  J


def predict(theta, X):
    '''calculates the accuracy given the inout data and the learned
    weihts'''
    p = sigmoid( X.dot(theta) ) >= 0.5
    return p


 




    

    
