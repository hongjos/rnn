import numpy as np

class Activation():
    """
    Base class skeleton for an activation function.
    """
    def __init__(self, type, eps=1e-10):
        """
        Initiliaze parameters
        ---
        type -> activation type
        eps -> for safe calculations
        """
        self.type = type
        self.eps = eps
    
    def get_name(self):
        """
        Get function name.
        """
        return self.type

    def __call__(self, x):
        """
        Computes the activation function applied to the input x.
        """
        raise NotImplemented("forward not implemented")

    def derivative(self, x):
        """
        Computes the derivative used for back propagation.
        """
        raise NotImplemented("derivative not implemented")

class Softmax(Activation):
    """
    Softmax activation class.
    """
    def __init__(self, type="Softmax", eps=1e-10):
        super().__init__(type, eps)

    def __call__(self, x):
        x = x + self.eps # for safety
        return np.exp(x) / np.sum(np.exp(x))

    def derivative(self, x):
        """
        Unused.
        """
        pass

class Sigmoid(Activation):
    """
    Sigmoid activation class.
    """
    def __init__(self, type="Sigmoid", eps=1e-10):
        super().__init__(type, eps)

    def __call__(self, x):
        x = x + self.eps # for safety
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        a = self(x)
        return a*(1 - a) # sigmoid derivative

class Tanh(Activation):
    """
    Tanh activation class
    """
    def __init__(self, type="Tanh", eps=1e-10):
        super().__init__(type, eps)

    def __call__(self, x):
        x = x + self.eps # for safety
        return np.tanh(x)

    def derivative(self, x):
        a = self(x)
        return 1 - a**2 # tanh derivative