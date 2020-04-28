import numpy as np

# Assignment 3 version

def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    loss = np.sum(W*W)*reg_strength
    grad = 2*W*reg_strength

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    batch_size = predictions.shape[0] if len(predictions.shape) > 1 else 1

    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)

    dprediction = softmax(predictions)
    dprediction[target_index if len(predictions.shape) == 1 else (range(batch_size), target_index)] -= 1
    dprediction /= batch_size

    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)
        
    def updateVal(self):
        self.value += self.grad

    def resetGrad(self):
        self.grad.fill(0)

        
def relu_func(x):
    return np.array([0 if i < 0 else i for i in x.flatten()]).reshape(x.shape)

def drelu_func(x):
    return np.array([0 if i < 0 else 1 for i in x.flatten()]).reshape(x.shape)


class ReLULayer:
    def relu_func(self, x):
        return 0 if x < 0 else x

    def drelu_func(self, x):
        return 0 if x < 0 else 1

    def __init__(self):
        #self.relu = np.vectorize(self.relu_func)
        #self.drelu = np.vectorize(self.drelu_func)
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X.copy()
        return relu_func(X)
        #res = np.vectorize(lambda a: 0 if a < 0 else a)(X)
        #return self.relu(X)

    def backward(self, d_out):
        # TODO copy from the previous assignment
        return d_out*drelu_func(self.X)

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X.copy()
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        # TODO copy from the previous assignment
        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.dot(np.ones((1, self.X.shape[0])), d_out)
        
        d_input = np.dot(d_out, self.W.value.T)
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = 0
        out_width = 0
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                pass
        raise Exception("Not implemented!")


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                pass

        raise Exception("Not implemented!")

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        raise Exception("Not implemented!")

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement backward pass
        raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}

    '''
    #Flattener from ther previous assignment:
    sample, label = data_train[0]
    print("SVHN data sample shape: ", sample.shape)
    # As you can see, the data is shaped like an image

    # We'll use a special helper module to shape it into a tensor
    class Flattener(nn.Module):
        def forward(self, x):
            batch_size, *_ = x.shape
            return x.view(batch_size, -1)
    '''
    
    