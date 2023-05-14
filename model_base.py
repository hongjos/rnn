import numpy as np

class Model:
    """
    Base class for the RNN Model. It's just an empty cell.
    """
    def __init__(self, input_dim, output_dim, hidden_dim, learning_rate, type):
        """
        Initialize the parameters with the input, output and hidden dimensions.
        ----
        input_dim -> dimension of input
        output_dim -> dimension of output 
        hidden_dim -> number of hidden units in a cell
        learning_rate -> learning rate for optimization
        type -> type of RNN used e.g many-to-one for classification, many-to-many for NER
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.type = type

        # initialize hidden state
        self.hidden = np.zeros((self.hidden_dim, 1))

        # initialize parameters and gradients (as dictionaries)
        self.P = {}
        self.G = {}

    def forward(self, X):
        """
        Does Nothing. This is where forward propagation should happen.
        ---
        X -> the input
        """
        return np.zeros((self.output_dim, 1))
    
    def backward(self, X, Y, Y_hat):
        """
        Does Nothing. This is where back propagation should happen.
        ----
        X -> the input passed in from the forward pass
        Y -> the expected output (of the input)
        Y_hat -> the predicted output (of the input)
        """
        return 0
    
    def optimize(self):
        """
        This is where the parameters are updated using the SGD optimizer.
        ----
        """
        # do one step
        for param in self.P:
            grad = 'd' + param
            self.P[param] -= self.learning_rate * self.G[grad]
    
    def fit(self, Xtrain, Ytrain, num_epochs, print_flag=False):
        """
        Fits the model with a given train dataset.
        Trains the model: forward prop, then back prop, and optimize.
        Returns a list of errors at each epoch.
        ----
        Xtrain -> inputs to be trained on
        Ytrain -> target values of inputs
        num_epochs -> how many times we loop through training
        print_flag -> prints the loss for every epoch if set to true
        """
        epoch_loss = [] # tracks the loss at each epoch

        # loop through training for each epoch
        for i in range(num_epochs):
            curr_loss = 0 # track loss at current epoch

            # iterate through each training example
            for j in range(len(Xtrain)):
                X = Xtrain[j] # current training example
                Y = Ytrain[j] # target of training example
                
                # forward pass
                Y_hat = self.forward(X)

                # back prop
                loss = self.backward(X, Y, Y_hat)

                # update parameters
                self.optimize()

                # update loss
                curr_loss += loss
            
            # store loss at epoch
            epoch_loss.append(curr_loss)
            if i % 5 == 0:
                print(f'Epoch {i}, training loss: {epoch_loss[-1]}')
        
        return epoch_loss
    
    def evaluate(self, Xtest, Ytest):
        """
        Evaluate the model based on a given test set. Returns the accruacy of the model.
        Assumes the model is many-to-one (classification).
        ----
        Xtest -> test inputs
        Yest -> test targets
        """
        num = len(Xtest)    # total number of test examples
        correct = 0         # number of correct predictions

        # iterate through each test example
        for i in range(num):
            X = Xtest[i] # current test example
            Y = Ytest[i] # target of test example

            # get prediction
            Y_pred = self.predict(X)
            
            # update number of correct examples if predicted matches target
            if Y_pred == np.argmax(Y):
                correct += 1
        
        # compute and return accuracy
        return correct / num
    
    def predict(self, X):
        """
        Predicts the output given an input based on the model parameters.
        ---
        X -> input vector
        """        
        # call forward
        self.forward(X)

        # return predicted output based on the type
        if self.type == "many-to-one":
            return np.argmax(self.outputs[-1]) # return which class
        else:
            return self.outputs
    
    def define_gradients(self):
        """
        Define the gradients for back propagation. 
        """
        # iterate through parameter names
        for param in self.P:
            # initialize gradient
            grad = 'd' + param
            self.G[grad] = np.zeros_like(self.P[param])
