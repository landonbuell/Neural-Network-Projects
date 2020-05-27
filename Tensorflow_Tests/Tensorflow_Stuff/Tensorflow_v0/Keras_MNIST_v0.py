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
import matplotlib.pyplot as plt

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    # LOAD IN DATASET
    data = keras.datasets.fashion_mnist
    (X_train,y_train) , (X_test,y_test) = data.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    n_classes = len(class_names)

    sample_idx = 100
    plt.title(class_names[y_train[sample_idx]],size=20)
    plt.imshow(X_train[sample_idx],cmap=plt.cm.binary)
    #plt.show()

    y_train = keras.utils.to_categorical(y_train,n_classes,dtype='uint8')
    y_test = keras.utils.to_categorical(y_test,n_classes,dtype='uint8')

    print(X_train.shape)

    # CREATE KERAS MLP MODEL
    MODEL = keras.Sequential(layers=[],name='JARVIS')
    MODEL.add(keras.layers.Flatten(input_shape=(28,28),name='Input'))
    MODEL.add(keras.layers.Dense(units=128,activation='relu',name='layer_1'))
    MODEL.add(keras.layers.Dense(units=10,activation='softmax',name='output'))  
    MODEL.compile(optimizer='sgd',loss='mse',
                  metrics=['Precision','Recall'])

    MODEL.fit(x=X_train,y=y_train,batch_size=128,epochs=200)