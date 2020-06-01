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

import tensorflow as tf
import tensorflow.keras as keras

import CNN_MNIST_Utilities as utils

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    # Set Checkpoint path
    exp_path = os.getcwd()
    print("\nModel parameters will be exported to\n",exp_path)

    # Load in Data Set
    print("\nLoading in MNIST Fashion data...")
    (X_train,y_train),(X_test,y_test) = utils.load_MNIST()

    # Build model
    print("\nConstructing Convolutional-Neural-Network (v1)...")
    CNN_v1 = utils.Network_Model('JARVIS')
    print(CNN_v1.summary())

    # Train & Show History of Model
    print("\tTraining",CNN_v1.name)
    model_history = CNN_v1.fit(x=X_train,y=y_train,batch_size=128,epochs=6)
    utils.plot_history(model_history,CNN_v1,show=True)


    # Export Model
    print("\tExporting",CNN_v1.name,"parameters...")
    local_path = exp_path+'/'+CNN_v1.name
    CNN_v1.save(filepath=local_path,save_format='tf')
    print("\t\tDone!")

    # Delete old Model
    print("\tDeleting",CNN_v1.name,"...")
    del(CNN_v1)
    print("\t\tDone :(")

    # Load Old Model
    print("\nLoading Old Model...")
    CNN_v2 = keras.models.load_model(filepath=local_path)
    print(CNN_v2.summary())

    # Resume Training
    print("\tResuming training...")
    model_history2 = CNN_v2.fit(x=X_train,y=y_train,batch_size=128,epochs=6)
    utils.plot_history(model_history2,CNN_v2,show=True)

    # Evaluate Model
    print("\nEvaluating:")
    results = CNN_v2.evaluate(x=X_train,y=y_train,batch_size=128,return_dict=True)
    print(results)