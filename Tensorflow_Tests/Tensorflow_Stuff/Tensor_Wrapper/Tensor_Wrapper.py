"""
Landon Buell
14 June 2020
"""

import numpy as np
import tensorflow as tf

import Wrapper_Utiitities as wrap_utils

if __name__ == '__main__':

    X_train,X_test,y_train,y_test = wrap_utils.Load_MNIST()

    MODEL = wrap_utils.Keras_Model(layers=[40,40],
                                   rows=np.arange(0,10),cols=np.arange(0,10))

