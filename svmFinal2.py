import os
import matplotlib
from random import shuffle, randint
from pylab import imshow
from scipy import io, ndimage
from numpy import *

from sklearn import svm, metrics

import matplotlib.pyplot as plt
from PIL import Image

def plotData(X, Y, c):
    '''plots the input data '''

    # plot any one case (20x20 image) from the input
    # the image matrix will have to be transposed to be viewed correcty
    inputImg = X[c,:]
    imshow((inputImg.reshape(20,20)).T, cmap = 'Greys')
    #ab = Image.fromarray(inputImg.reshape(20,20))
    #ab.show()
    
    #plot the same ouptut case
    print('the digit printed is', Y[c][0] )



def plotData2(image):
    '''plots the input data '''
    # the image matrix will have to be transposed to be viewed correcty
    # cmap shows the color map
    plt.imshow(image.T, cmap='Greys')
    plt.show()



def arrangeData(X, Y, cutoff):
    '''divides the data into training, cross validation and test datsets '''
    
    # get input matrix shape and no. of output classes
    n, m = X.shape
    k = unique(Y).size

    # examples per output-class
    epc = int(n/k)
    epcTrain = int(epc*cutoff)
    epcTest = epc - epcTrain - 1
    
    # choosing training and test dataset size
    trainDataSize = n*cutoff
    testDataSize = n - trainDataSize
    
    # initializing training and test dataset
    XTrain = zeros((trainDataSize, m))
    XTest = zeros((testDataSize , m))

    YTrain = zeros(trainDataSize)    
    YTest = zeros(testDataSize)

    # thresholding the dataset
    #X = (X-amin(X))/(amax(X)-amin(X))
    #X[X>0.5] = 1
    #X[X<=0.5] = 0

    # assigning examples to training dataset
    for i in range(0,k):
        for j in range(0, epcTrain):
            XTrain[i*epcTrain + j] = X[i*epc + j]
            YTrain[i*epcTrain + j] = Y[i*epc + j]

    # assigning exampples to test dataset
    for i in range(0,k):
        for j in range(0, epcTest):
            XTest[i*epcTest + j] = X[i*epc + j+epcTrain]
            YTest[i*epcTest + j] = Y[i*epc + j+epcTrain]

    return XTrain, YTrain, XTest, YTest

#=========================== arrange data for use ==========================

# load the data
data = io.loadmat('database/num2020.mat')
size = (20,20)

# making X and Y numpy arrays
X = data['X']
Y = data['Y']

# changing identity of digits
Y = Y-1

# getting the no. of examples and size of each example
numOfExamples, sizeOfExample = X.shape

# getting the no. of different classes in the output    
numOfLabels = unique(Y).size   

# choosing training , cross validation and test dataset size
cutoff = 0.80
XTrain, YTrain, XTest, YTest = arrangeData(X, Y, cutoff)

# plotting 
#print('plotting a random digit from the input')
randomIndex = randint(0, X.shape[0])
#plotData(X, Y, randomIndex)
#plt.show()

image = X[randomIndex,:]
image = image.reshape(20,20)
#plotData2(image)



# ================== set params for plotting graph =========================

# gamma and C values. change it acc to your data
gammaValue = 0.01
cValue = 10

# ====================== training and testing =============================

#creating a Support vector classifier
classifier = svm.SVC(C = cValue , kernel='rbf', gamma = gammaValue , tol=0.1)
       
# learning on the training dataset
classifier.fit(XTrain, YTrain)

# predicting on training and test dataset
predictionTest = classifier.predict(XTest)
predictionTrain = classifier.predict(XTrain)

# calculating efficiencies
effTest = 100 * metrics.accuracy_score(YTest, predictionTest, normalize = True)
effTrain = 100 * metrics.accuracy_score(YTrain, predictionTrain, normalize = True)

print('training set eff =', effTrain, '\t test set eff=', effTest)

# ================== post training and testing work ========================

# testing on a random input
print('testing on a random input from test set')
c = randint(0, XTest.shape[0])

predictedOutput = classifier.predict(XTest[c])[0]
actualOutput = YTest[c]
print('predicted output is', predictedOutput, 'and actual output is', actualOutput)

# ============== testing on unseen images out of dataset ====================

imageName = ['01.png', '02.png' , '03.png', '04.png' , '05.png',
             '11.png', '12.png' , '13.png', '14.png' , '15.png',
             '21.png', '22.png' , '23.png', '24.png' , '25.png',
             '31.png', '32.png' , '33.png', '34.png' , '35.png',
             '41.png', '42.png' , '43.png', '44.png' , '45.png',
             '51.png', '52.png' , '53.png', '54.png' , '55.png',
             '61.png', '62.png' , '63.png', '64.png' , '65.png',
             '71.png', '72.png' , '73.png', '74.png' , '75.png',
             '81.png', '82.png' , '83.png', '84.png' , '85.png',
             '91.png', '92.png' , '93.png', '94.png' , '95.png']

print('testing on a png image')
for x in imageName:

    fullname = os.path.join('testImages/',x)
    im = Image.open(fullname)

    if(len(shape(im))==3):
        imA = asarray(im, dtype='float')[:,:,1]
    else:
        imA = asarray(im, dtype='float')

    # transform pixel vaues from 0-1
    imA = (imA-amin(imA))/(amax(imA) - amin(imA))
    imA = 1-imA

    im1 = asarray(imA, dtype='float')
    im1 = ndimage.grey_dilation(im1, size = (25,25))
    im1 = Image.fromarray(im1)

    #obtain boundingbox
    box = (im1).getbbox()
    im2 = im1.crop(box)

    im3 = im2.resize(size)
    im3 = asarray(im3, dtype="float")
    im3 = 1- im3.T

    im3 = uint8(im3)
    #plotData2(im3)
    im3 = im3.reshape((1,X.shape[1]))

    #plotData(im3, y, 0)
    #plt.show()

    predictedOutput = classifier.predict(im3[0])[0]
    print('predicted output is', predictedOutput)
    
    
