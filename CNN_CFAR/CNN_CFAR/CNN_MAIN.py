"""
Landon Buell
CNN Tests
CFAR Dataset - MAIN
22 June 2020
"""

            #### IMPORTS ####

import numpy as np
import CNN_Utilities as CNN_utils

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

        # PREPROCESS DATA
        X_train,y_train,X_test,y_test = CNN_utils.load_CFAR10()

        print("X train shape:",X_train.shape)
        print("y train shape:",y_train.shape)
        print("X test shape:",X_test.shape)
        print("y train shape:",y_test.shape)

        # CREATE & TRAIN MODEL
        MODEL = CNN_utils.Create_Network()

        MODEL.fit(x=X_train,y=y_train,batch_size=128,
                  epochs=10,werbose=2,validation_split=0.1)