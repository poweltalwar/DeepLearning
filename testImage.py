import os
import sys
from numpy import *
from scipy import io
from scipy import ndimage
import matplotlib.pyplot as plt
from pybrain.structure import *
from pybrain.datasets import SupervisedDataSet
from pybrain.utilities import percentError
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml.networkreader import NetworkReader
from PIL import Image


def plotData(image):
    '''plots the input data '''
    # the image matrix will have to be transposed to be viewed correcty
    # cmap shows the color map
    plt.imshow(image.T, cmap='Greys')
    plt.show()


# load data    
data = io.loadmat('database/num4040.mat')
size = (40, 40)
X = data['X']
Y = data['Y']
    
# read test image
im = Image.open("testImages/01.png")

# convert to numpy array
if(len(shape(im)) == 3):
    imA = asarray(im, dtype="float")[:,:,1]
else:
    imA = asarray(im, dtype="float")

# transform pixel values from 0 to 1 and invert and convert to PIL image
imA = (imA - amin(imA)) / (amax(imA) - amin(imA))
imA = 1 - imA
#im1 = Image.fromarray(imA)

# obtain bounding box for image and crop
im1 = asarray(imA, dtype="float")
im1 = ndimage.grey_dilation(im1, size=(25,25))

im1 = Image.fromarray(im1)
box = (im1).getbbox()
im2 = im1.crop(box)

# resize this cropped bounding box, convert to numpy array
#im2 = asarray(im2, dtype="float")
#im21 = asarray(im2, dtype="float")
#im2 = ndimage.grey_dilation(im21, size=(7,7))
im3 = im2.resize(size)
im3 = asarray(im3, dtype="float")
#im2[im2 > .5] = 1
#im2[im2 < .5] = 0

im3 = 1 - im3.T
im3 = uint8(im3)
plotData(im3)

#im = im / amax(im)
#im[im >= .5] = 1
#im[im < .5] = 0


X1 = im3.reshape((X.shape[1]))
#X1 = reshape(X1, (len(X1)))

net = NetworkReader.readFrom('solutions/netTrain4040.xml')
prediction = net.activate(X1)
net.activate(X1)
p = argmax(prediction, axis=0)
print(prediction)
print("predicted output is \t" + str(p))
