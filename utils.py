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

def mult_zeros_like(params):
    """
    Same thing as numpy's zeros like except it works with lists.
    Used for defining/initializing gradients.
    """
    return [np.zeros_like(p) for p in params]

def clip_gradients(grads, val=1):
    """
    Clips gradients to avoid exploding gradients.
    ---
    value -> maximum gradient value
    """
    return [np.clip(g, -val, val) for g in grads]