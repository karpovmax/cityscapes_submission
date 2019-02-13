#!/usr/bin/python3
from __future__ import print_function, absolute_import, division
import os, glob, sys
from skimage.io import imread, imshow, imsave
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np
import matplotlib.pyplot as plt


def getData(num_tests, start, type): 

    if 'CITYSCAPES_DATASET' in os.environ:
        cityscapesPath = os.environ['CITYSCAPES_DATASET']
    else:
        cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
    searchAnnotated = os.path.join(cityscapesPath, "gtFine", type, "*", "*_gt*_labelTrain*")
    searchRaw = os.path.join(cityscapesPath, "x_test", "*", "*")

    #if not searchAnnotated:
    #    printError("Did not find any annotated files.")
    filesAnnotated =glob.glob(searchAnnotated)

    filesRaw=glob.glob(searchRaw)
    filesAnnotated.sort()
    filesRaw.sort()
    #print (len(filesAnnotated))
    return [], filesRaw[start:start+num_tests]

def UpscaleImg(img,scale, dims):
    if dims:
        new_img = np.zeros((img.shape[0]*scale,img.shape[1]*scale,3))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                new_img[i*scale:(i+1)*scale,j*scale:(j+1)*scale,:]=img[i,j,:]
    else:
        new_img = np.zeros((img.shape[0] * scale, img.shape[1] * scale))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                new_img[i * scale:(i + 1) * scale, j * scale:(j + 1) * scale] = img[i, j]
    return new_img

def importBatch(num_tests, start, verbose, type="train", scale=1):   #load batch of data from train dataset

    y_files, X_files = getData(num_tests,start, type)
    X_input = []
    y_input = []
   # if type=='val':
    filenames = []
    z = 0
    for i in range(len(X_files)):

        z+=1
        if verbose:
            if z % 100 == 0:
                print('loaded files input - ', z)

        X_file = X_files[i]
        filenames.append(X_file[:-16])
        X_img = imread(X_file)

        if (scale != 0):
            X_new = np.zeros((int(X_img.shape[0] / scale), int(X_img.shape[1] / scale),3))
            k = 0
            for x in X_img[::scale]:
                X_new[k]=x[::scale]
                k+=1
                X_img = X_new
        X_input.append(X_img)
    z = 0
    for i in range(len(y_files)):
        z += 1
        if verbose:
            if z % 100 == 0:
               print('loaded files output - ', z)

        y_file = y_files[i]
        y_img = imread(y_file)
        if (scale != 0):
            y_new = np.zeros((int(y_img.shape[0] / scale), int(y_img.shape[1] / scale)))
            k = 0
            for y in y_img[::scale]:
                y_new[k] = y[::scale]
                k += 1
                y_img = y_new
        y_input.append(y_img)


    X = np.array(X_input)
    y = np.array(y_input)
    if (type=='val' or type=='test'):
        return X,y, filenames
    return X, y



IMG_SHAPE = (1024, 2048, 3)
EPOCHS = 20
BATCH_SIZE = 1
TOTAL_SIZE = 200
EPOCH_SIZE = int(TOTAL_SIZE / BATCH_SIZE)
VAL_SIZE = 40
SCALE_RATE = 1
VERBOSE = 1
START_INDEX = 0


import os, glob, sys
import matplotlib.pyplot as plt
import numpy as np
import h5py
import tensorflow as tf
#%matplotlib inline
import skimage
from skimage.io import imread, imshow, imsave
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import time
import functools
#from eval import *


import random
#%env CITYSCAPES_DATASET = /home/skim/data/
from tensorflow.metrics import *
#%load_ext autoreload
#%autoreload 2
def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta  = 0.5

    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that pixels are class i
    p1 = ones - y_pred # proba that pixels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0, (0, 1, 2))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2)) + beta * K.sum(p1 * g0, (0, 1, 2)) + 1e-8

    T = K.sum(num / den) # when summing over classes, T has dynamic range [0 Ncl]

    classNumber = K.cast(K.shape(y_true)[-1], 'float32') ### equal classNumber = 20.0
    return classNumber - T



def eval_model(model):
    x_pred = model.predict(x_val_data, verbose=VERBOSE)
    new_x = np.argmax(x_pred, axis=3)
    new_x = new_x.astype(int)
    y_val = np.argmax(y_val_data, axis=3)
    score = eval_preds(new_x, y_val)
    return score

#xTest = "../rvygon_data"
output_dir = "res"
xTest, output_dir = sys.argv[1:]
os.environ['CITYSCAPES_DATASET'] = xTest
x_test, yyyyyy, filenames = importBatch(500, 0, 0, 'test', 1) 

x_test = x_test.astype('uint8') 
with tf.device('/cpu:0'): #device:GPU:1
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = load_model('program/unet_140epochs.hdf5', custom_objects={'tversky_loss': tversky_loss})
        #sess.run(tf.global_variables_initializer())
        pred = model.predict(x_test, verbose=0)
        pred = np.argmax(pred,axis=3).astype(int)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for i in range(len(filenames)): 
            impath = os.path.join(output_dir, filenames[i].split('/')[-1]+'_gtFine_labelIds.png')
            imsave(impath, pred[i])
