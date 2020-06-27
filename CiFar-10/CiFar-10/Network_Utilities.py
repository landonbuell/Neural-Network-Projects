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

            #### VARIABLE DECLARATIONS ####

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

            #### FUNCTION DEFINITIONS ####

def Neural_Network (name):
    """ Create Untrained Keras Sequential Model """
    model = keras.models.Sequential(name=name)
    model.add(keras.layers.InputLayer(input_shape=(32,32,3),name='Input'))

    # Convolution / Pooling layers
    model.add(keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='C1'))
    model.add(keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='C2'))
    model.add(keras.layers.MaxPool2D(pool_size=(4,4),name='P1'))

    #model.add(keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_uniform',name='C1'))
    #model.add(keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_uniform',name='C2'))
    #model.add(keras.layers.MaxPool2D(pool_size=(4,4),name='P1'))

    #model.add(keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_uniform',name='C1'))
    #model.add(keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_uniform',name='C2'))
    #model.add(keras.layers.MaxPool2D(pool_size=(4,4),name='P1'))

    # Dense layers
    model.add(keras.layers.Flatten(name='F1'))
    model.add(keras.layers.Dense(units=64,activation='relu',kernel_initializer='he_uniform',name='D1'))
    model.add(keras.layers.Dense(units=10,activation='softmax',name='Output'))
    # Compile
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['Precision','Recall'])
    print(model.summary())
    return model

def Plot_History (hist,model,save=False,show=True):
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

