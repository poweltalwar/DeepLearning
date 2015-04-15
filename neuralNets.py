import os
import sys
from numpy import *
from scipy import io
import matplotlib.pyplot as plt
from pybrain.structure import *
from pybrain.datasets import SupervisedDataSet
from pybrain.utilities import percentError
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from PIL import Image


def plotData(image):
    '''plots the input data '''
    # the image matrix will have to be transposed to be viewed correcty
    # cmap shows the color map
    plt.imshow(image, cmap='Greys')
    plt.show()
    

def convertToOneOfMany(Y):
    '''converts supervised dataset to softmax classifier'''
    rows, cols = shape(Y)
    numLabels = len(unique(Y))

    Y2 = zeros((rows, numLabels))
    for i in range(0, rows):
        Y2[i, Y[i]] = 1

    return Y2


# load data    
data = io.loadmat('database/num20201.mat')
size = (20, 20)
X = data['X']
[m, n] = shape(X)
Y = data['Y']
Y = reshape(Y, (len(Y), -1))

numLabels = len(unique(Y))
Y[Y == 10] = 0

# bias term added
#X = hstack((ones((m, 1)), X))
#n = n+1

# threshold the images 
#X[X < 0.5] = 0
#X[X >= .5] = 1


# set sizes of layers
nInput = n
nHidden0 = int(n)
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
dataTrain, dataTest = allData.splitWithProportion(.9)

# create object for training
train = BackpropTrainer(net, dataset=dataTrain, learningrate=0.03, momentum=0.3)
#train.trainUntilConvergence(dataset=dataTrain)

# evaluate correct output for trainer
trueTrain = dataTrain['target'].argmax(axis=1)
trueTest = dataTest['target'].argmax(axis=1)

# train step by step
EPOCHS = 60

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


outTrain = net.activateOnDataset(dataTrain)
outTrain = outTrain.argmax(axis=1)
resTrain = 100 - percentError(outTrain, trueTrain)

outTest = net.activateOnDataset(dataTest)
outTest = outTest.argmax(axis=1)
resTest = 100 - percentError(outTest, trueTest)

print("epoch: %4d " % train.totalepochs,"\ttrain acc: %5.2f%% " % resTrain,
          "\ttest acc: %5.2f%%" % resTest)

NetworkWriter.writeToFile(net, 'solutions/netTrain20207.xml')
