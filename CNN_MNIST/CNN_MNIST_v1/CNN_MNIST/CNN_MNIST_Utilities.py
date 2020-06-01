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

def load_MNIST ():
    """
    Load in MNIST Fashin Data for training/testing
    --------------------------------
    (no args)
    --------------------------------
    Return split design matrix & target arrays
    """
    fashion_MNIST = keras.datasets.fashion_mnist
    (X_train,y_train),(X_test,y_test) = fashion_MNIST.load_data()

    X_train = X_train.reshape(60000,28,28,1)
    X_test = X_test.reshape(10000,28,28,1)

    n_classes = np.unique(y_train).shape[0]
    y_train = keras.utils.to_categorical(y_train,n_classes)
    y_test = keras.utils.to_categorical(y_train,n_classes)

    return (X_train,y_train),(X_test,y_test)


def Network_Model (name):
    """
    Construct Convolutional Nerual Network Model
    --------------------------------
    name (str) : Name for Network object
    --------------------------------
    Return Compiled Network Model
    """
    model = keras.models.Sequential(name=name)
    model.add(keras.layers.InputLayer(input_shape=(28,28,1)))

    model.add(keras.layers.Conv2D(filters=64,kernel_size=(6,6),strides=(2,2),activation='relu'))
    model.add(keras.layers.AveragePooling2D(pool_size=(2,2),strides=None))
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(units=40,activation='relu'))
    model.add(keras.layers.Dense(units=10,activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(),
                  loss='categorical_crossentropy',
                  metrics=[keras.metrics.Precision(),keras.metrics.Recall()])

    return model

def plot_history (hist,model,save=False,show=False):
    """
    Visualize Data from Keras History Object Instance
    --------------------------------
    hist (inst) : Keras history object
    --------------------------------
    Return None
    """
    # Initialize Figure

    eps = np.array(hist.epoch)          # arr of epochs
    n_figs = len(hist.history.keys())

    fig,axs = plt.subplots(nrows=n_figs,ncols=1,sharex=True,figsize=(20,8))
    plt.suptitle(model.name+' History',size=50,weight='bold')
    hist_dict = hist.history
    
    for I in range (n_figs):                # over each parameter
        key = list(hist_dict)[I]
        axs[I].set_ylabel(str(key).upper(),size=20,weight='bold')
        axs[I].plot(eps,hist_dict[key])     # plot key
        axs[I].grid()                       # add grid

    plt.xlabel("Epochs",size=20,weight='bold')

    if save == True:
        plt.savefig(title.replace(' ','_')+'.png')
    if show == True:
        plt.show()
