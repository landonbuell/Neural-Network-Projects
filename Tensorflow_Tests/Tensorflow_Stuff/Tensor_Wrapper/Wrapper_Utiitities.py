"""
Landon Buell
14 June 2020
"""

import numpy as np
import bitstring
import tensorflow as tf
import tensorflow.keras as keras

        #### CLASS DEFINITIONS ####

class ApproximationLayer (keras.layers.Layer):
    """
    Create layer to Apply Approximate computations method to.
    --------------------------------
    rows (iter) : Array-like of rows to apply approximations to
    cols (iter) : Array-like of cols to apply approximations to
    --------------------------------
    Returns initiated Approximation Layer Instance
    """

    def __init__(self,rows=[0],cols=[0]):
        """ Initialize Approximation Layer Object """
        super(ApproximationLayer,self).__init__(trainable=False)
        self.rows = rows        # row indexs to apply MSB
        self.cols = cols        # rol index to apply MSB
        return None

    def mute(self,x):
        """ Mute bit in single FP-64 value """
        print(type(x))

    def Mute_MSB (self,X):
        """ Mute Most-Significant-bit in samples,rows,cols """
        for i in range(len(X)):
            for r in self.rows:
                for c in self.cols:
                    X[i][r][c] = self.mute(X[i][r][c])
        return X

    def call (self,inputs):
        """ Define Compution from input to produce outputs """
        output = self.Mute_MSB(inputs)
        return inputs

            #### FUNCTION DEFINITIONS ####

def Load_MNIST ():
    """ Collect Training & Testing Data from keras.datasets """
    print("Collecting MNIST data .....\n")
    (X_train,y_train),(X_test,y_test) = \
        keras.datasets.mnist.load_data()
    X_test,y_test = X_test[:6000],y_test[:6000]
    X_train,y_train = X_train[:10000],y_train[:10000]
    X_train,X_test = X_train/255,X_test/255
    return X_train,X_test,y_train,y_test

def Keras_Model (layers,rows=[],cols=[]):
    """
    Create Keras Sequential Model
    --------------------------------
    layers (tup) : Iterable with i-th elem is units in i-th Dense Layer
    rows (iter) : Array-like of rows to apply approximations 
    cols (iter) : Array-like of cols to apply approximations 
    --------------------------------
    Return untrained, Compiled Keras Model
    """
    model = keras.models.Sequential(name='Digit_Classifier')
    model.add(keras.layers.Input(shape=(100,28,28),name='Image'))
    #model.add(ApproximationLayer(rows=rows,cols=cols))
    model.add(keras.layers.Flatten())
    for I,neurons in enumerate(layers):
        model.add(keras.layers.Dense(units=neurons,activation='relu',
                                     name='Hidden_'+str(I)))
    model.add(keras.layers.Dense(units=10,activation='softmax',name='Output'))
    model.compile(optimizer=keras.optimizers.SGD(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['Precision','Recall'])
    print(model.summary())
    return model

