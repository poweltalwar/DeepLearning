import os
import sys
from numpy import *
from scipy import optimize
from pylab import *
from prog2 import *


#import the data
data  = loadtxt('ex2data1.txt', delimiter=',')

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
X_train = hstack((X0,X_train))

#initialise the fitting parametrers
initial_theta = zeros( (n + 1, 1) );

#initialise lambda for regularisation
Lambda = .3;

#compute initial cost and gradient
cost = costFunction(initial_theta, X_train, Y_train, Lambda)

options = {'full_output': True, 'maxiter': 400}


theta, cost, _, _, _ =  optimize.fmin(lambda t: costFunction(t, X_train, Y_train, Lambda ), initial_theta, **options)

p = predict(theta, X_pre)

print('Train Accuracy:', (p == Y_pre).mean() * 100)
