import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features row

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = X # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs @ self.weights_input_to_hidden) # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = hidden_outputs # signals into final output layer
        final_outputs = final_inputs @ self.weights_hidden_to_output # signals from final output layer
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # This page was a huge help in figuring this out: http://neuralnetworksanddeeplearning.com/chap2.html

        # TODO: Output error - Replace this value with your calculations.
        error = 1/2 * (y - final_outputs)**2
        error_deriv = final_outputs - y
        activation_prime = 1 # derivative of our final layer activation f(x) is just 1
        output_error_term = error_deriv * activation_prime

        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error_deriv = self.weights_hidden_to_output @ output_error_term
        sigma_prime = hidden_outputs * (1 - hidden_outputs)
        hidden_error_term = hidden_error_deriv * sigma_prime

        # Weight step (input to hidden)
        # shapes: (i,) @ ((h,) * (h,))
        # reshape so we end up with (i, h) to match delta_weights_i_h
        delta_weights_i_h -= X[:, None] @ hidden_error_term[None, :]
        # Weight step (hidden to output)
        # shapes: (h,) @ ((o,) * (o,))
        # reshape so we end up with (h,o) to match delta_weights_h_o
        delta_weights_h_o -= hidden_outputs[:, None] @ output_error_term[None, :]
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''

        final_outputs, _ = self.forward_pass_train(features)
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
# 1800 / 0.1  / 70  Training loss: 0.371 ... Validation loss: 0.792
# 1800 / 0.1  / 150 Training loss: 0.631 ... Validation loss: 1.103
# 2000 / 0.05 / 70  Training loss: 0.357 ... Validation loss: 0.659
# 2000 / 0.05 / 50  Training loss: 0.358 ... Validation loss: 0.637
iterations = 5200
learning_rate = 0.5
hidden_nodes = 18
# doesn't really make sense to change this for the assignment itself
output_nodes = 1
