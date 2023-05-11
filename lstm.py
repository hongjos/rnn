import numpy as np
from activation import *
from utils import *
from model_base import Model

class LSTM(Model):
    """
    A standard recurrent neural network (RNN) model.
    """
    def __init__(self, input_dim, output_dim, hidden_dim, learning_rate=1e-5, type='many-to-one'):
        """
        Initialize the RNN.
        """
        super().__init__(input_dim, output_dim, hidden_dim, learning_rate, type)

        self.ih_dim = input_dim + hidden_dim # input + hidden dimension
        dlen = np.sqrt(self.ih_dim)               # used to normalize weights
        
        # initialize parameters
        W_f = np.random.randn(hidden_dim, self.ih_dim) / dlen   # weight matrix (forget gate)
        W_i = np.random.randn(hidden_dim, self.ih_dim) / dlen   # weight matrix (input/update gate) 
        W_m = np.random.randn(hidden_dim, self.ih_dim) / dlen   # weight matrix (modulation gate/candidate) 
        W_o = np.random.randn(hidden_dim, self.ih_dim) / dlen   # weight matrix (output gate) 
        W_h = np.random.randn(output_dim, hidden_dim) / dlen    # weight matrix (hidden to output)

        b_f = np.zeros((hidden_dim, 1)) # bias (forget gate) 
        b_i = np.zeros((hidden_dim, 1)) # bias (update gate)
        b_m = np.zeros((hidden_dim, 1)) # bias (modulation gate/candidate)
        b_o = np.zeros((hidden_dim, 1)) # bias (output gate)
        b_h = np.zeros((output_dim, 1)) # bias (hidden to output)

        # initialize the hidden state and memory
        self.hidden = np.zeros((self.hidden_dim, 1)) 
        self.cmem = np.zeros((self.hidden_dim, 1))       

        # the activation function (just use softmax for now)
        self.activation = Softmax()

        # the loss function used (cross entropy loss)
        self.loss_function = CELoss()
        
    def feed_forward(self, X):
        """
        Computes the forward pass of the LSTM.
        Returns the output at the last step.
        """
        ### initialize storage for the components in the LSTM ###
        # combined hidden and input states
        self.hi_states = []

        # forget, input/update, modualation, output gate states
        self.f_states, self.i_states, self.m_states, self.o_states = [], [], [], []

        # hidden and memory states
        self.hidden_states, self.c_states = [], []

        # hidden to output, and output states
        self.h_states, self.outputs = [], []

        # add previous hidden and memory to state
        self.hidden_states.append(self.hidden)
        self.c_states.append(self.cmem)

        tanh = Tanh()                   # tanh activation
        t_range = len(X)                # number of elements in input

        # iterate through each element in the input vector
        for t in range(t_range):
           x = 1 
        
        return self.outputs[-1]
    
    def back_prop(self, X, Y, Y_hat):
        """
        Computes the gradients through back propagation, returning the loss.
        For classifaction (many-to-one) we only care about the last output.
        """
        loss = 0        # keep track of loss
        tanh = Tanh()   # tanh activation

        # define gradients
        self.define_gradients() 
        da_next = np.zeros_like(self.hidden_states[0]) # keeps track of derivative for next hidden state
        
        # for classifaction (many-to-one) we only care about the final output
        # so the output loss is the same across all cells
        dy = Y_hat.copy() # loss gradient
        
        if self.type == 'many-to-one':
            loss = self.loss_function.forward(Y, Y_hat)
            # compute the gradient of the loss w.r.t output
            dy = self.loss_function.backward()
        
        # go through hidden layers and update gradients
        # we ignore the initial (t=0) hidden layer
        for t in reversed(range(len(self.hidden_states))):
            x = 1

        # clip gradients
        grads = [x]
        [x] = clip_gradients(grads)

        return loss

    def optimize(self):
        """
        This is where the parameters are updated using the SGD optimizer.
        ----
        """   
        params = []
        grads = []

        # do one step
        for i in range(len(params)):
            params[i] -= self.learning_rate * grads[i]

        # make sure parameters are updated
        [] = params
    
    def define_gradients(self):
        """
        Define the gradients for back propagation. 
        """
        params = []

        x = mult_zeros_like(params)
        
        