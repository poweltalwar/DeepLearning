import os
import sys
from numpy import *
from scipy import optimize
from pylab import *
from prog2 import *


#import the data
data  = loadtxt('ex2data1.txt', delimiter=',')

numrows = len(data)
numcols = len(data[0])

#separate the data
X = data[:,0:numcols-1]
Y = data[:,numcols-1]

#add the bias term to data
m = numrows
n = numcols-1
X0 = ones((m,1))
X = hstack((X0,X))

#initialise the fitting parametrers
initial_theta = zeros( (n + 1, 1) );

#compute initial cost and gradient
cost = costFunction(initial_theta, X, Y)

options = {'full_output': True, 'maxiter': 400}

theta, cost, _, _, _ =  optimize.fmin(lambda t: costFunction(t, X, Y), initial_theta, **options)

p = predict(theta, X)

print('Train Accuracy:', (p == Y).mean() * 100)
