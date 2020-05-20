"""
Landon Buell
Nueral Network Projects
Chapter 4
30 April 2020
"""


            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import os
import random

if __name__ == '__main__':

    # Get list of file names
    path = 'C:/Users/Landon/Documents/Python_Stuff/PetImages/Cat'
    _, _, cat_images = next(os.walk(path))

    # Visualize 3  x 3 images
    fig,ax = plt.subplots(nrows=3,ncols=3,figsize=(12,12))

    for idx,img in enumerate(random.sample(cat_images,9)):
        img_read = plt.imread(path+img)
        ax[int(idx/3),idx%3].imshow(img_read)
        ax[int(idx/3),idx%3].axis('off')
        ax[int(idx/3),idx%3].set_title("Cat"+img)
    plt.show()