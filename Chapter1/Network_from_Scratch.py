"""
Landon Buell
Nerual Network Projects with Python
Chapter 1 - Network from Scratch
26 April 2020
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt

            #### DEFINITIONS ####

def sigmoid (z):
    """ Sigmoid Activation Functions """
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv (z):
    """ Derivative of Sigoid Activation Functions """
    return z * (1. - z)

class Neural_Network ():
    """ Create a Neural Network Instance """

    def __init__ (self,x,y):
        """ Initilize Network Instance """
        self.input = x
        self.y = y
        self.weights1 = np.random.rand(self.input.shape[1],4)
        self.weights2 = np.random.rand(4,1)
        self.output = np.zeros(self.y.shape)

    def feed_forward (self):
        """ Forward pass data through network """
        self.layer1 = sigmoid(np.dot(self.input,self.weights1))
        self.output = sigmoid(np.dot(self.layer1,self.weights2))

    def back_prop (self):
        """ Backward propagate throught network """

        # compute gradients
        d_weights2 = np.dot(self.layer1.T,2*(self.y-self.output)*
                            sigmoid_deriv(self.output))
        d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y-self.output) * 
                            sigmoid_deriv(self.output),self.weights2.T) *
                            sigmoid_deriv(self.layer1)))
        # update weights w/ gradients
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def loss_value (self):
        """ Comput loss value at current iteration """
        return np.sum((self.output - self.y )**2)

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':
    """ Run Neural Network """

    # Training data set
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[1],[0]])

    # Create network  and list for loss
    NN = Neural_Network(X,y)
    losses = np.array([])
    n_iters = 1500

    # train for N iters
    for i in range (n_iters):
        NN.feed_forward()
        losses = np.append(losses,NN.loss_value())
        NN.back_prop()

    print(NN.output)

    # Plot Loss function
    plt.figure(figsize=(12,8))
    plt.title("Network Loss",size=40,weight='bold')
    plt.xlabel("Training Iteration",size=20,weight='bold')
    plt.ylabel("Loss Function Value",size=20,weight='bold')
    plt.plot(np.arange(0,n_iters),losses,
             color='blue',label='Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    
