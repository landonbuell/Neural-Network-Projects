"""
Landon Buell
Keras Test
Geron, Ch.9
21 May 2020
"""

            #### Imports ####

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow import keras

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    

    # CREATE KERAS MODEL
    inputs = keras.Input(shape=(784,),batch_size=100,name='Digits')
    x = keras.layers.Dense(units=128,activation='relu',name='Dense_1')(inputs)
    x = keras.layers.Dense(units=64,activation='relu',name='Dense_2')(x)
    outputs = keras.layers.Dense(units=10,activation='softmax',name='Output')(x)

    MODEL = keras.Model(inputs=inputs,outputs=outputs)

    # INITIALIZE DATA SET
    (X_train,y_train),(y_train,y_test) = keras.datasets.mnist.load_data()

    X_train = X_train.reshape(60000,784,).astype(tf.float32)
    X_test = X_test.reshape(10000,784,).astype(tf.float32)
    y_train = y_train.astype(tf.uint8)
    y_test = y_test.astype(tf.uint8)

    # CONFIGURATION
    MODEL.compile(optimizer='sgd',loss='mse')

    # TRAINING ON DATA:
    history = MODEL.fit(x=X_train,y=y_train,batch_size=100,epochs=2)

    hos
