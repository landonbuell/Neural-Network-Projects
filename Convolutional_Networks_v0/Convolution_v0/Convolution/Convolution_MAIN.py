"""
Landon Buell
Convolutions v0
Main
21 May 2020
"""

        #### MAIN EXECUTABLE ####

import numpy as np
import tensorflow as tf

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


import Convolution_Utilities as conv_utils


if __name__ == '__main__':

    MNIST = fetch_openml('mnist_784',version=1)
    X,y = MNIST['data'],MNIST['target'].astype(np.uint16)
    X_train,X_test,y_train,y_test = \
        train_test_split(X,y,test_size=0.4)

    #feature_cols = tf.contrib.infer_real_valued_columns_from_input(X_train)