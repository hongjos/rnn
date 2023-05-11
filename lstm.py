import numpy as np
from activation import *
from utils import *
from model_base import Model

class LSTM(Model):
    """
    A standard long short-term memort (LSTM) RNN model.
    """
    def __init__(self, input_dim, output_dim, hidden_dim, learning_rate=1e-3, type='many-to-one'):
        """
        Initialize the LSTM.
        """
        super().__init__(input_dim, output_dim, hidden_dim, learning_rate, type)

        self.ih_dim = input_dim + hidden_dim # input + hidden dimension
        dlen = np.sqrt(self.ih_dim)          # used to normalize weights
        
        # initialize parameters
        self.W_f = np.random.randn(hidden_dim, self.ih_dim) / dlen   # weight matrix (forget gate)
        self.W_i = np.random.randn(hidden_dim, self.ih_dim) / dlen   # weight matrix (input/update gate) 
        self.W_c = np.random.randn(hidden_dim, self.ih_dim) / dlen   # weight matrix (candidate gate) 
        self.W_o = np.random.randn(hidden_dim, self.ih_dim) / dlen   # weight matrix (output gate) 
        self.W_y = np.random.randn(output_dim, hidden_dim) / dlen    # weight matrix (hidden to output)

        self.U_f = np.random.randn(hidden_dim, hidden_dim) / dlen   # weight matrix recurrence (forget)
        self.U_i = np.random.randn(hidden_dim, hidden_dim) / dlen   # weight matrix recurrence (input) 
        self.U_c = np.random.randn(hidden_dim, hidden_dim) / dlen   # weight matrix recurrence (candidate) 
        self.U_o = np.random.randn(hidden_dim, hidden_dim) / dlen   # weight matrix recurrence (output gate) 

        self.b_f = np.zeros((hidden_dim, 1)) # bias (forget gate) 
        self.b_i = np.zeros((hidden_dim, 1)) # bias (update gate)
        self.b_c = np.zeros((hidden_dim, 1)) # bias (candidate)
        self.b_o = np.zeros((hidden_dim, 1)) # bias (output gate)
        self.b_y = np.zeros((output_dim, 1)) # bias (hidden to output)

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
        # re-initialize hidden and cell state
        self.hidden = np.zeros((self.hidden_dim, 1))
        self.cmem = np.zeros((self.hidden_dim, 1))
        
        ### initialize storage for the components in the LSTM ###
        # combined hidden and input states
        self.x_states = []

        # forget, input/update, candidate, output gate states
        self.f_states, self.i_states, self.c_states, self.o_states = [], [], [], []

        # hidden and memory states
        self.hidden_states, self.cmem_states = [], []

        # hidden to output, and output states
        self.y_states, self.outputs = [], []

        # add previous hidden and memory to state
        self.hidden_states.append(self.hidden)
        self.cmem_states.append(self.cmem)

        tanh = Tanh()                   # tanh activation
        sigmoid = Sigmoid()             # sigmoid activation
        t_range = len(X)                # number of elements in input

        # iterate through each element in the input vector
        for t in range(t_range):
            # concatenate hidden and input state
            x = np.row_stack((self.hidden, X[t]))
            self.x_states.append(x)
            
            # compute forget gate
            forget = sigmoid.forward(np.dot(self.W_f, x) + np.dot(self.U_f, self.hidden) + self.b_f)

            # compute input/update gate   
            update = sigmoid.forward(np.dot(self.W_i, x) + np.dot(self.U_i, self.hidden) + self.b_i)

            # compute candidate  
            cand = tanh.forward(np.dot(self.W_c, x) + np.dot(self.U_c, self.hidden) + self.b_c)

            # compute output gate        
            out = sigmoid.forward(np.dot(self.W_o, x) + np.dot(self.U_o, self.hidden)+ self.b_o)      

            # compute new memory
            self.cmem = forget*self.cmem + update*cand 
            
            # compute new hidden state
            self.hidden = out * tanh.forward(self.cmem)

            # compute the hidden to output state
            h_o = np.dot(self.W_y, self.hidden) + self.b_y

            # compute the prediction
            y = self.activation.forward(h_o)

            # store computations
            self.f_states.append(forget)
            self.i_states.append(update)
            self.c_states.append(cand)
            self.o_states.append(out)
            self.cmem_states.append(self.cmem)
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
        dcmem_next = np.zeros_like(self.cmem_states[0]) 
        
        # for classifaction (many-to-one) we only care about the final output
        # so the output loss is the same across all cells
        dy = Y_hat.copy() # loss gradient
        
        if self.type == 'many-to-one':
            loss = self.loss_function.forward(Y, Y_hat)
            # compute the gradient of the loss w.r.t output
            dy = self.loss_function.backward()
        
        # go through hidden layers and update gradients
        for t in reversed(range(len(self.hidden_states[1:]))):
            # if many-to-many type RNN we compute the loss at each step
            if self.type == "many-to-many":
                loss += self.loss_function.forward(Y[t], self.outputs[t])
                # compute the gradient of the loss w.r.t output
                dy = self.loss_function.backward()
            
            # update gradient for hidden to output
            self.dW_y += np.dot(dy, self.hidden_states[t+1].T)
            self.db_y += dy

            # compute derivative for hidden and output states       
            dh = np.dot(self.W_y.T, dy) + dhidden_next
            do = sigmoid.backward(self.o_states[t])*dh*tanh.forward(self.cmem_states[t])

            # update gradients for output gate
            self.dW_o += np.dot(do, self.x_states[t].T)
            self.dU_o += np.dot(do, self.hidden_states[t].T)
            self.db_o += do

            # compute derivative for the cell memory state and candidate
            dcmem = np.copy(dcmem_next) + dh*self.o_states[t]*tanh.backward(tanh.forward(self.cmem_states[t]))
            dc = dcmem*self.i_states[t]
            dc = tanh.backward(self.c_states[t])*dc
            
            # update the gradients with respect to the candidate
            self.dW_c += np.dot(dc, self.x_states[t].T)
            self.dU_c += np.dot(dc, self.hidden_states[t].T)
            self.db_c += dc

            # update gradients for input/update gate
            di = sigmoid.backward(self.i_states[t])*dcmem*self.c_states[t]
            self.dW_i += np.dot(di, self.x_states[t].T)
            self.dU_i += np.dot(di, self.hidden_states[t].T)
            self.db_i += di

            # update gradients for forget fate
            df = sigmoid.forward(self.f_states[t])*dcmem*self.cmem_states[t-1]
            self.dW_f += np.dot(df, self.x_states[t].T)
            self.dU_f += np.dot(df, self.hidden_states[t].T)
            self.db_f += df

            # update gradients for next hidden cell mem state
            dhi = (np.dot(self.W_f.T, df) + np.dot(self.W_i.T, di) + np.dot(self.W_c.T, dc)+ np.dot(self.W_o.T, do))
            dhidden_next = dhi[:self.hidden_dim, :]
            dcmem_next = self.f_states[t]*dcmem

        # clip gradients
        grads = [self.dW_f, self.dW_i, self.dW_c, self.dW_o, self.dW_y,
                 self.dU_f, self.dU_i, self.dU_c, self.dU_o,
                 self.db_f, self.db_i, self.db_c, self.db_o, self.db_y]
        [self.dW_f, self.dW_i, self.dW_c, self.dW_o, self.dW_y,
         self.dU_f, self.dU_i, self.dU_c, self.dU_o,
         self.db_f, self.db_i, self.db_c, self.db_o, self.db_y] = clip_gradients(grads)

        return loss

    def optimize(self):
        """
        This is where the parameters are updated using the SGD optimizer.
        ----
        """   
        params = [self.W_f, self.W_i, self.W_c, self.W_o, self.W_y,
                  self.U_f, self.U_i, self.U_c, self.U_o,  
                  self.b_f, self.b_i, self.b_c, self.b_o, self.b_y]
        grads = [self.dW_f, self.dW_i, self.dW_c, self.dW_o, self.dW_y,
                 self.dU_f, self.dU_i, self.dU_c, self.dU_o,
                 self.db_f, self.db_i, self.db_c, self.db_o, self.db_y]

        # do one step
        for i in range(len(params)):
            params[i] -= self.learning_rate * grads[i]

        # make sure parameters are updated
        [self.W_f, self.W_i, self.W_c, self.W_o, self.W_y,
         self.U_f, self.U_i, self.U_c, self.U_o,
         self.b_f, self.b_i, self.b_c, self.b_o, self.b_y] = params
    
    def define_gradients(self):
        """
        Define the gradients for back propagation. 
        """
        params = [self.W_f, self.W_i, self.W_c, self.W_o, self.W_y,
                  self.U_f, self.U_i, self.U_c, self.U_o,  
                  self.b_f, self.b_i, self.b_c, self.b_o, self.b_y]
        
        [self.dW_f, self.dW_i, self.dW_c, self.dW_o, self.dW_y,
         self.dU_f, self.dU_i, self.dU_c, self.dU_o, 
         self.db_f, self.db_i, self.db_c, self.db_o, self.db_y] = mult_zeros_like(params)
        
        