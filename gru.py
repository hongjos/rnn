import numpy as np
from activation import *
from utils import *
from model_base import Model

class GRU(Model):
    """
    A standard gated recurrent unit (GRU) RNN model.
    """
    def __init__(self, input_dim, output_dim, hidden_dim, learning_rate=1e-3, type='many-to-one'):
        """
        Initialize the GRU.
        """
        super().__init__(input_dim, output_dim, hidden_dim, learning_rate, type)

        # set parameter names
        self.param_names = ['W_r', 'W_u', 'W_c', 'W_y',
                            'U_r', 'U_u', 'U_c',
                            'b_r', 'b_u', 'b_c', 'b_y']
        
        # set gradient names
        self.grad_names = [add_char(p, 'd') for p in self.param_names]

        dlen = np.sqrt(self.hidden_dim) # used to normalize weights
        
        # initialize parameters
        self.P['W_r'] = np.random.randn(hidden_dim, input_dim) / dlen   # weight matrix (reset gate)
        self.P['W_u'] = np.random.randn(hidden_dim, input_dim) / dlen   # weight matrix (update gate) 
        self.P['W_c'] = np.random.randn(hidden_dim, input_dim) / dlen   # weight matrix (candidate gate) 
        self.P['W_y'] = np.random.randn(output_dim, hidden_dim) / dlen  # weight matrix (hidden to output)

        self.P['U_r'] = np.random.randn(hidden_dim, hidden_dim) / dlen  # weight matrix recurrence (reset)
        self.P['U_u'] = np.random.randn(hidden_dim, hidden_dim) / dlen  # weight matrix recurrence (update) 
        self.P['U_c'] = np.random.randn(hidden_dim, hidden_dim) / dlen  # weight matrix recurrence (candidate) 

        self.P['b_r'] = np.zeros((hidden_dim, 1)) # bias (reset gate) 
        self.P['b_u'] = np.zeros((hidden_dim, 1)) # bias (update gate)
        self.P['b_c'] = np.zeros((hidden_dim, 1)) # bias (candidate)
        self.P['b_y'] = np.zeros((output_dim, 1)) # bias (hidden to output)

        # initialize the hidden state
        self.hidden = np.zeros((self.hidden_dim, 1))      

        # the activation function (just use softmax for now)
        self.activation = Softmax()

        # the loss function used (cross entropy loss)
        self.loss_function = CELoss()
        
    def feed_forward(self, X):
        """
        Computes the forward pass of the GRU.
        Returns the output at the last step.
        """
        # re-initialize hidden state
        self.hidden = np.zeros((self.hidden_dim, 1))

        ### initialize storage for the components in the GRU ###
        # reset, update, candidate, hidden to output gate states
        self.r_states, self.u_states, self.c_states, self.y_states = [], [], [], []

        # hidden states
        self.hidden_states, self.outputs = [], []

        tanh = Tanh()                   # tanh activation
        sigmoid = Sigmoid()             # sigmoid activation
        t_range = len(X)                # number of elements in input

        # iterate through each element in the input vector
        for t in range(t_range):
            # compute reset gate
            reset = sigmoid.forward(np.dot(self.P['W_r'], X[t]) + np.dot(self.P['U_r'], self.hidden) + self.P['b_r'])

            # compute update gate      
            update = sigmoid.forward(np.dot(self.P['W_u'], X[t]) + np.dot(self.P['U_u'], self.hidden) + self.P['b_u'])

            # compute candidate     
            cand = tanh.forward(np.dot(self.P['W_c'], X[t]) + np.dot(self.P['U_c'], self.hidden) +  self.P['b_c'])          

            # compute new hidden state
            self.hidden = np.multiply(update, self.hidden) + np.multiply((1 - update), cand)

            # compute the hidden to output state
            h_o = np.dot(self.P['W_y'], self.hidden) + self.P['b_y']

            # compute the prediction
            y = self.activation.forward(h_o)

            # store computation
            self.r_states.append(reset)
            self.u_states.append(update)
            self.c_states.append(cand)
            self.hidden_states.append(self.hidden)
            self.y_states.append(h_o)
            self.outputs.append(y)
        
        return self.outputs[-1]
    
    def back_prop(self, X, Y, Y_hat):
        """
        Computes the gradients through back propagation, returning the loss.
        For classifaction (many-to-one) we only care about the last output.
        """
        loss = 0            # keep track of loss
        tanh = Tanh()       # tanh activation
        sigmoid = Sigmoid() # sigmoid activation

        # define gradients
        self.define_gradients()

        # keeps track of gradient for next hidden state and memory
        dhidden_next = np.zeros_like(self.hidden_states[0])  
        
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
            
            # update gradient for hidden to output
            self.G['dW_y'] += np.dot(dy, self.hidden_states[t].T)
            self.G['db_y'] += dy

            # compute derivatives for hidden and candidate
            dh = np.dot(self.P['W_y'].T, dy) + dhidden_next
            dc = tanh.backward(self.c_states[t])*np.multiply(dh, (1 - self.u_states[t]))
            
            # update gradients for candidate gate
            self.G['dW_c'] += np.dot(dc, X[t].T)
            self.G['dU_c'] += np.dot(dc, np.multiply(self.r_states[t], self.hidden_states[t-1]).T)
            self.G['db_c'] += dc
            
            # compute derivatives for reset gate
            dr = np.multiply(np.dot(self.P['U_c'].T, dc), self.hidden_states[t-1])
            dr = dr*sigmoid.backward(self.r_states[t])
            
            # update reset gate gradients
            self.G['dW_r'] += np.dot(dr, X[t].T)
            self.G['dU_r'] += np.dot(dr, self.hidden_states[t-1].T)
            self.G['db_r'] += dr
            
            # compute derivatives for update gate
            du = np.multiply(dh, self.hidden_states[t-1] - self.c_states[t])
            du = du*sigmoid.backward(self.u_states[t])
            
            # update update gradients
            self.G['dW_u'] += np.dot(du, X[t].T)
            self.G['dU_u'] += np.dot(du, self.hidden_states[t-1].T)
            self.G['db_u'] += du
            
            # update next hidden state gradient
            dhnext = np.dot(self.P['U_u'].T, du) + np.multiply(dh, self.u_states[t])
            dhnext += np.multiply(np.dot(self.P['U_c'].T, dc), self.r_states[t]) + np.dot(self.P['U_r'].T, dr)

        # clip gradients
        clip_gradients(self.G)

        return loss 
        