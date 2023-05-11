import numpy as np

class CELoss:
    """
    Class for the cross entropy loss function. 
    """
    def __init__(self):
        self.type = 'CELoss'
    
    def forward(self, Y, Y_hat):
        """
        Computes the cross entropy loss for an expected out.
        ----
        Y -> the expected output
        Y_hat -> the predicted output
        """
        self.Y = Y
        self.Y_hat = Y_hat

        loss = np.sum(-Y*np.log(Y_hat), axis=0).mean()
        return loss

    def backward(self):
        """
        Computes the derivative of the loss for back prop.
        """
        return self.Y_hat - self.Y

def clip_gradients(grads, val=1):
    """
    Takes in a dictionary of gradients and clips them to avoid exploding gradients.
    ----
    value -> maximum gradient value
    """
    for g in grads:
        grads[g] = np.clip(g, -val, val)

def add_char(name, char):
    """
    Adds a single character to the front of a string. Used for naming gradients.
    ----
    char -> the character to be added
    """
    return char + name