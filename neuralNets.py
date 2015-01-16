import os
import sys
from numpy import *
from numpy import random
from scipy import io
import pylab
from matplotlib import *
import matplotlib.pyplot as plt
from pybrain.structure import *
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet, SupervisedDataSet
from pybrain.utilities import percentError
from pybrain.supervised.trainers import BackpropTrainer


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


def splitData(alldata, factor):
    '''shuffles the data and divides into 2 parts with sizes
    decided by ratio'''
        
    #shuffle data
    #shuffle(allData)

    #calculate new sizes
    numrows = len(data)
    numrow_train = int(numrows*factor)

    #separate the data
    data1 = data[0:numrow_train,:]
    data2 = data[numrow_train:numrows,:]

    return data1, data2


def convertToOneOfMany(Y):
    rows, cols = shape(Y)
    numLabels = len(unique(Y))

    Y2 = zeros((rows, numLabels))
    for i in range(0, rows):
        Y2[i, Y[i]] = 1

    return Y2


    
data = io.loadmat('ex4data1.mat')
X = data['X']
[m, n] = shape(X)
#X = X[950:1050,:]
Y = data['y']
Y = reshape(Y, (len(Y), -1))
#Y = Y[950:1050,:]
numLabels = len(unique(Y))
Y[Y == 10] = 0 # '0' is encoded as '10' in data, fix it

#set sizes of layers
nInput = n
nHidden0 = n
nOutput = numLabels

#define layer structures
inLayer = LinearLayer(n)
hiddenLayer = SigmoidLayer(10) #set asone for each class or one for each input feature
outLayer = SoftmaxLayer(numLabels)

#add layers to network
net = FeedForwardNetwork()
net.addInputModule(inLayer)
net.addModule(hiddenLayer)
net.addOutputModule(outLayer)

#define conncections for network
theta1 = FullConnection(inLayer, hiddenLayer)
theta2 = FullConnection(hiddenLayer, outLayer)

#add connections to network
net.addConnection(theta1)
net.addConnection(theta2)

#sort module
net.sortModules()

#activate by giving input
#n.activate(X[0])

allData = SupervisedDataSet(n, 10)
Y2 = convertToOneOfMany(Y)

for i in range(m):
    inData = X[i,:]
    outData = Y2[i, :]
    allData.addSample(inData, outData)

#allData.setField('input', X)
#allData.setField('target', Y)

#separate training and testing data
dataTrain, dataTest = allData.splitWithProportion(0.70)

#dataTrain._convertToOneOfMany()
#dataTest._convertToOneOfMany()

#train = BackpropTrainer(net, learningrate=0.01, momentum=0.9, verbose=True)
train = BackpropTrainer(net, dataset=dataTrain)
#train.trainUntilConvergence(dataset=dataTrain)
'''train.trainEpochs(50)
resTrain = percentError( train.testOnClassData ( dataset=dataTest ),True )
resTest = percentError( train.testOnClassData ( dataset=dataTrain ),True )
'''

for i in range(50):
    train.trainEpochs(1)
    trnresult = percentError(train.testOnClassData(dataset=dataTest), True)
    tstresult = percentError(train.testOnClassData(dataset=dataTrain), True)
    print("epoch: %4d" % train.totalepochs,"  train error: %5.2f%%" % trnresult,
              "  test error: %5.2f%%" % tstresult)

testNum = plotData(X, Y)
X1 = X[testNum, :]
prediction = net.activate(X1)
print(prediction)
p = argmax(prediction, axis=0)
print("predicted output is \t" + str(p))
