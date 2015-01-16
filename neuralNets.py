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


def plotData(X, Y, c):
    '''plots the input data '''
    # plot any one case (20x20 image) from the input
    # the image matrix will have to be transposed to be viewed correcty
    # cmap shows the color map
    m, n = shape(X)
    image = array(X[c,1:n])
    plt.imshow((image.reshape(20, 20)).T, cmap='Greys')
    plt.show()
    #plot the same ouptut case
    print("true number is " + str(Y[c]))


def convertToOneOfMany(Y):
    '''converts supervised dataset to softmax classifier'''
    rows, cols = shape(Y)
    numLabels = len(unique(Y))

    Y2 = zeros((rows, numLabels))
    for i in range(0, rows):
        Y2[i, Y[i]] = 1

    return Y2


# load data    
data = io.loadmat('ex4data1.mat')
X = data['X']
[m, n] = shape(X)
Y = data['y']
Y = reshape(Y, (len(Y), -1))

numLabels = len(unique(Y))
Y[Y == 10] = 0

# bias term added
X = hstack((ones((m, 1)), X))
n = n+1

# set sizes of layers
nInput = n
nHidden0 = int(n / 5)
nOutput = numLabels

# define layer structures
inLayer = LinearLayer(nInput)
hiddenLayer = SigmoidLayer(nHidden0)
outLayer = SoftmaxLayer(nOutput)

# add layers to network
net = FeedForwardNetwork()
net.addInputModule(inLayer)
net.addModule(hiddenLayer)
net.addOutputModule(outLayer)

# define conncections for network
theta1 = FullConnection(inLayer, hiddenLayer)
theta2 = FullConnection(hiddenLayer, outLayer)

# add connections to network
net.addConnection(theta1)
net.addConnection(theta2)

# sort module
net.sortModules()

# testing befor training
c = random.randint(0, X.shape[0])
print("testing without training \nchoosing a random number\t" +
      str(c))
X1 = X[c, :]
prediction = net.activate(X1)
net.activate(X1)
p = argmax(prediction, axis=0)
plotData(X, Y, c)
#print("true output is\t" + str(Y[c]))
print("predicted output is \t" + str(p))

# create a dataset object, make output Y a softmax matrix
allData = SupervisedDataSet(n, numLabels)
Y2 = convertToOneOfMany(Y)

# add data samples to dataset object, both ways are correct
'''for i in range(m):
    inData = X[i,:]
    outData = Y2[i, :]
    allData.addSample(inData, outData)
'''
allData.setField('input', X)
allData.setField('target', Y2)

#separate training and testing data
dataTrain, dataTest = allData.splitWithProportion(0.70)

# create object for training
train = BackpropTrainer(net, dataset=dataTrain, learningrate=0.1, momentum=0.1)
#train.trainUntilConvergence(dataset=dataTrain)

# evaluate correct output for trainer
trueTrain = dataTrain['target'].argmax(axis=1)
trueTest = dataTest['target'].argmax(axis=1)

# train step by step
EPOCHS = 20
for i in range(EPOCHS):
    train.trainEpochs(1)

    # accuracy on training dataset
    outTrain = net.activateOnDataset(dataTrain)
    outTrain = outTrain.argmax(axis=1)
    resTrain = 100 - percentError(outTrain, trueTrain)

    #accuracy on test dataset
    outTest = net.activateOnDataset(dataTest)
    outTest = outTest.argmax(axis=1)
    resTest = 100 - percentError(outTest, trueTest)
    
    print("epoch: %4d " % train.totalepochs,"\ttrain acc: %5.2f%% " % resTrain,
          "\ttest acc: %5.2f%%" % resTest)


prediction = net.activate(X1)
print(prediction)
p = argmax(prediction, axis=0)
print("predicted output is \t" + str(p))
