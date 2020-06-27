"""
Landon Buell
CiFar-10
Test Program
27 June 2020
"""

            #### IMPORTS ####

import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras

import Network_Utilities as utils

             #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    (X_train,y_train),(X_test,y_test) = keras.datasets.cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train,10)
    X_train = X_train / 255
    X_test = X_test / 255

    MODEL = utils.Neural_Network('VISION')

    history = MODEL.fit(x=X_train,y=y_train,batch_size=128,epochs=10,verbose=1)

    utils.Plot_History(history,MODEL)

