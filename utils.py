import numpy as np
from activation import *

class CELoss:
    """
    Class for the cross entropy loss function. 
    """
    def __init__(self):
        self.type = 'CELoss'
        self.type = Softmax()
    
    def __call__(self, Y, Y_hat):
        """
        Computes the cross entropy loss for an expected out.
        ----
        Y -> the expected output
        Y_hat -> the predicted output
        """
        self.Y = Y
        self.Y_hat = Y_hat

        # compute the cross entropy loss
        loss = np.sum(-Y*np.log(Y_hat), axis=1).mean()
        return loss

    def derivative(self):
        """
        Computes the derivative of the loss for back prop.
        """
        dy = self.Y_hat.copy()
        dy[np.argmax(self.Y)] -= 1
        return dy # derivative of softmax CE loss

def clip_gradients(grads, val=1):
    """
    Takes in a dictionary of gradients and clips them to avoid exploding gradients.
    ----
    value -> maximum gradient value
    """
    # go through gradients and clip them
    for g in grads:
        grads[g] = np.clip(grads[g], -val, val)

def add_char(name, char):
    """
    Adds a single character to the front of a string. Used for naming gradients.
    ----
    char -> the character to be added
    """
    return char + name
