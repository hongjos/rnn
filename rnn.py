import numpy as np
from activation import *
from utils import *
from model_base import Model

class RNN(Model):
    """
    A standard recurrent neural network (RNN) model.
    """
    def __init__(self, input_dim, output_dim, hidden_dim, learning_rate=.1, type='many-to-one'):
        """
        Initialize the RNN.
        """
        super().__init__(input_dim, output_dim, hidden_dim, learning_rate, type)

        # set parameter names
        self.param_names = ['W_ax', 'W_aa', 'W_ya', 'b_a', 'b_y']
        
        # set gradient names
        self.grad_names = [add_char(p, 'd') for p in self.param_names]

        dlen = np.sqrt(hidden_dim) # used to normalize weights
        
        # initialize parameters
        self.P['W_ax'] = np.random.randn(hidden_dim, input_dim) / dlen    # weight matrix (input to hidden state)
        self.P['W_aa'] = np.random.randn(hidden_dim, hidden_dim) / dlen   # weight matrix (recurrent, hidden to hidden)
        self.P['W_ya'] = np.random.randn(output_dim, hidden_dim) / dlen   # weight matrix (hidden to output)
        
        self.P['b_a'] = np.zeros((hidden_dim, 1))                         # bias (hidden)
        self.P['b_y'] = np.zeros((output_dim, 1))                         # bias (output)

        # initialize hidden state
        self.hidden = np.zeros((self.hidden_dim, 1))        

        # the activation function (just use softmax for now)
        self.activation = Softmax()

        # the loss function used (cross entropy loss)
        self.loss_function = CELoss()
        
    def forward(self, X):
        """
        Computes the forward pass of the RNN.
        Returns the output at the last step.
        """
        # re-initialize hidden state
        self.hidden = np.zeros((self.hidden_dim, 1))

        # initialize storage for hidden states and output
        self.hidden_states, self.outputs = [], [] 

        tanh = Tanh()                   # tanh activation
        t_range = len(X)                # number of elements in input

        # iterate through each element in the input vector
        for t in range(t_range):
            # compute new hidden state
            self.hidden = tanh.forward(np.dot(self.P['W_aa'], self.hidden) + np.dot(self.P['W_ax'], X[t]) + self.P['b_a'])

            # compute output
            y = self.activation.forward(np.dot(self.P['W_ya'], self.hidden) + self.P['b_y'])

            # store output and hidden state
            self.hidden_states.append(self.hidden.copy())
            self.outputs.append(y)
        
        return self.outputs[-1]
    
    def backward(self, X, Y, Y_hat):
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
        for t in reversed(range(len(self.hidden_states))):
            # if many-to-many type RNN we compute the loss at each step
            if self.type == "many-to-many":
                loss += self.loss_function.forward(Y[t], self.outputs[t])
                # compute the gradient of the loss w.r.t output
                dy = self.loss_function.backward()

            # update gradients for output weights and bias
            self.G['dW_ya'] += np.dot(dy, self.hidden_states[t].T)
            self.G['db_y'] += dy

            # compute gradient for hidden state
            da = np.dot(self.P['W_ya'].T, dy) + da_next 

            # update gradients for hidden weights and bias
            dtanh = tanh.backward(self.hidden_states[t])*da
            self.G['db_a'] += dtanh
            self.G['dW_ax'] += np.dot(dtanh, X[t].T)
            self.G['dW_aa'] += np.dot(dtanh, self.hidden_states[t-1].T)

            # update gradient for next hidden state
            da_next = np.dot(self.P['W_aa'].T, dtanh)

        # clip gradients
        clip_gradients(self.G)

        return loss
            