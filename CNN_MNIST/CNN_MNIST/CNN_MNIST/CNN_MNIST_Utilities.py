"""
Landon Buell
CNN Test
Utiulities Script
30 May 2020
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow.keras as keras


            #### FUNCTIONS DEFINITIONS ####

def Network_Model (name):
    """
    Construct Convolutional Nerual Network Model
    --------------------------------
    name (str) : Name for Network object
    --------------------------------
    Return Compiled Network Model
    """
    model = keras.models.Sequential(name=name)
    model.add(keras.layers.InputLayer(input_shape=(None,28,28)))

    model.add(keras.layers.Conv2D(filters=2,kernel_size=(6,6),
                                  strides=(2,2)))
    model.add(keras.layers.AveragePooling2D(2,2),strides=None)

    model.add(keras.layers.Conv2D(filters=1,kernel_size=(3,3),
                                  strides=(1,1)))
    model.add(keras.layers.MaxPool1D(poolsize=2,strides=None))

    model.add(keras.layers.Dense(units=100,activation='relu'))
    model.add(keras.layers.Dense(units=40,activation='relu'))

    model.add(keras.layers.Dense(units=10,activation='softmax'))
    model.compile(optimizer='sgd',loss='categorical_crossentropy',
                  metrics=['precision','recall'])

    return model




