"""
Landon Buell
Nerual Network Projects with Python
Chapter 1 - Network with Keras
26 April 2020
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':
    """ Create Neural Network with Keras API """

    # create instance
    model = Sequential()

    # add dense layers
    model.add(Dense(units=4,activation='sigmoid',input_dim=3))
    model.add(Dense(units=1,activation='sigmoid'))
    print(model.summary())

    # SGD optimizer
    SGD = optimizers.SGD(learning_rate=1)
    model.compile(optimizer=SGD,loss='mean_squared_error')

    # Training data set
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[1],[0]])

    # Run predictions
    n_iters = 1500
    model.fit(X,y,epochs=n_iters,verbose=False)
    print(model.predict(X))