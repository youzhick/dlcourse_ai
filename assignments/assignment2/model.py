import numpy as np
import layers
import assignment1.linear_classifer as lc

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        hidden_layer_size, int - number of neurons in the hidden layer
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        reg, float - L2 regularization strength
        """

        self.reg = reg
        # TODO Create necessary layers
        self.n_input = n_input
        self.n_output = n_output
        self.layers = [
            FullyConnectedLayer(n_input, hidden_layer_size),
            ReLULayer(),
            FullyConnectedLayer(hidden_layer_size, n_output)
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!

        for p in self.params().values():
            p.resetGrad()
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        x = X.copy()
        for l in self.layers:
            x = l.forward(x)
        
        loss, dpred = lc.softmax_with_cross_entropy(x, y)

        for l in reversed(self.layers):
            dpred = l.backward(dpred)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for p in self.params().values():
            l2loss, gr = lc.l2_regularization(p.value, self.reg)
            loss += l2loss
            p.grad += gr

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        x = X.copy()
        for l in self.layers:
            x = l.forward(x)
        pred = np.argmax(x, axis=1)

        return pred
    
    def params(self):
        result = {}

        # TODO Implement aggregating all of the params
        for i in range (len(self.layers)):
            pars = self.layers[i].params()
            for p in pars:
                result[p + str(i)] = pars[p]

        return result
