"""
Landon Buell
CNN Tests
CFAR Dataset _Utils
22 June 2020
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf
import tensorflow.keras as keras

            #### VARIABLES ####

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

            #### DEFINITITIONS ####

def load_CFAR10 ():
    """ Load CFAR 10 Dataset """
    (X_train,y_train),(X_test,y_test) = keras.datasets.cifar10.load_data()
    return X_train,y_train,X_test,y_test

def Create_Network (inshape=(None,32,32,3),n_classes=10):
    """ Create keras Convolutional Neural Network """

    model = keras.models.Sequential(name='VISION')
    model.add(keras.layers.Input(shape=inshape,name='Input'))
    model.add(keras.layers.Conv2D(filters=64,kernelsize=(3,3),name='C1'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2),name='P1'))
    model.add(keras.layers.Conv2D(filters=32,kernelsize=(3,3),name='C2'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2),name='P2'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=20,activation='relu',name='D1'))
    model.add(keras.layers.Dense(units=n_classes,activation='softmax',name='Output'))

    model.compile(optimizier=keras.optimizers.SGD(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['Precision','Recall'],)
    print(model.summary())
    return model
