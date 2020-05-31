"""
Landon Buell
CNN Test
MAIN Script
30 May 2020
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow.keras as keras

import CNN_MNIST_Utilities as utils

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    # Load in Data Set
    fashion_MNIST = keras.datasets.fashion_mnist
    (X_train,y_train),(X_test,y_test) = fashion_MNIST.load_data()

    n_classes = np.unique(y_train).shape[0]
    y_train = keras.utils.to_categorical(y_train,n_classes)

    CNN_MODEL = utils.Network_Model('JARVIS')
    history = CNN_MODEL.fit(x=X_train,y=y_train,batch_size=128,epochs=100)