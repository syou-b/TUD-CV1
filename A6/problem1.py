import numpy as np

class Network:
    """
    Constructs a neural network according to a list detailing the network structure.

    Args:
        network_structure: List of integers, number of nodes in a layer. The entry network_structure[0]
        equals the number of input features and the last entry is 1 for binary classification.
        network_structure[i] equals the number of hidden units in layer i.
    """

    def __init__(self, network_structure):
        self.num_layers = len(network_structure) - 1
        # state dicts, use integer layer id as keys
        # the range is 0,...,num_layers for x and
        # 1,...,num_layers for all other dicts
        self.w = dict() # weights
        self.b = dict() # biases
        self.z = dict() # outputs of linear layers
        self.x = dict() # outputs of activation layers
        self.dw = dict() # error derivatives w.r.t. w
        self.db = dict() # error derivatives w.r.t. b

        self.init_wb(network_structure)


    def init_wb(self, network_structure):
        """ Initialize all parameters w[i] and b[i] for i = 1,..., num_layers of the neural network.
        Tip: If the current weight of the neuron layer is a matrix of n_i x n_(i-1), i is the 
        network layer number, and n is the number of nodes in the network layer represented by the 
        subscript i.
        For example, [3,2,4,1] is a 3-layer structure: the input size is 3, the
        number of neurons in the hidden layer 1 is 2, the number of neurons in the hidden layer 2 is 4
        and the output size of layer 3 is 1.
        Initialize weight matrices randomly from a normal distribution with variance 1 / n_(i-1).
        Initialize biases to 0.
        """
        #
        # You code here
        #
        for i in range(1, self.num_layers + 1): # 1, 2, ..., num_layers
            # layer: N_(i-1) -> N_i
            self.w[i] = np.random.normal(0, 1/network_structure[i-1], (network_structure[i], network_structure[i-1])) # sigma, size
            self.b[i] = np.zeros((network_structure[i], 1), dtype = float) # (n_i, 1)


    def sigmoid(self, z):
        """ Sigmoid function.

        Args:
            z: (n_i, b) numpy array, output of the i-th linear layer

        Returns:
            x_out: (n_i, b) numpy array, output of the sigmoid function
        """
        #
        # You code here
        #
        return np.array([(1 / (1 + np.e ** (-i))) for i in z])


    def sigmoid_backward(self, dx_out, z):
        """ Backpropagation for the sigmoid function.

        Args:
            dx_out: (n_i, b) numpy array, partial derivatives dE/dx_i of error with respect to 
                    the output of the sigmoid function in layer i
            z: (n_i, b) numpy array, output of the i-th linear layer

        Returns:
            dz_in: (n_i, b) numpy array, partial derivatives dE/dz_i of the error with respect to
                the output of the i-th linear layer
        """
        #
        # You code here
        #
        # dE/dz_i = dE/dx_i * dx_i/dz_i (chain rule)
        #         = dx_out * dx_i/dz_i
        # (n_i, b) = (n_i, b) * (n_i, b) (element-wise multiplication)
        sigmoid_derivative = 1 / (2 + np.e ** z + np.e ** (-z)) # dSigmoid(x)/dx
        return np.multiply(dx_out, sigmoid_derivative) # element-wise multiplication


    def relu(self, z):
        """ ReLU function.

        Args:
            z: (n_i, b) numpy array, output of the i-th linear layer

        Returns:
            x_out: (n_i, b) numpy array, output of the ReLU function
        """
        #
        # You code here
        #
        return np.maximum(0, z)
    

    def relu_backward(self, dx_out, z):
        """ Backpropagation for the ReLU function.

        Args:
            dx_out: (n_i, b) numpy array, partial derivatives dE/dx_i of error with respect to 
                    the output of the ReLU function in layer i
            z: (n_i, b) numpy array, output of the i-th linear layer

        Returns:
            dz_in: (n_i, b) numpy array, partial derivatives dE/dz_i of the error with respect to
                the output of the i-th linear layer
        """
        #
        # You code here
        #
        # dE/dz_i = dE/dx_i * dx_i/dz_i (chain rule)
        #         = dx_out * dx_i/dz_i
        # (n_i, b) = (n_i, b) * (n_i, b) (element-wise multiplication)
        relu_derivative = np.where(z <= 0, 0, 1) # dx_i/dz_i = 0 if z <=0 else 1
        return np.multiply(dx_out, relu_derivative) # element-wise multiplication


    def activation_func(self, func, z):
        """ Select and perform forward pass through activation function.

        Args:
            func: string, either "sigmoid" or "relu"
            z: (n_i, b) numpy array, output of the i-th linear layer

        Returns:
            x_out: (n_i, b) numpy array, output of the ReLU function
        """
        if func == "sigmoid":
            return self.sigmoid(z)
        elif func == "relu":
            return self.relu(z)


    def activation_func_backward(self, func, dx_out, z):
        """ Select and perform backward pass through activation function.

        Args:
            func: string, either "sigmoid" or "relu"
            dx_out: (n_i, b) numpy array, partial derivatives dE/dx_i of error with respect to 
                    the output of the activation function in layer i
            z: (n_i, b) numpy array, output of the i-th linear layer

        Returns:
            dz_in: (n_i, b) numpy array, partial derivatives dE/dz_i of the error with respect to
                the output of the i-th linear layer
        """
        if func == "sigmoid":
            return self.sigmoid_backward(dx_out, z)
        elif func == "relu":
            return self.relu_backward(dx_out, z)


    def layer_forward(self, x_in, func, i):
        """ Forward propagation through the i-th network layer.
        Uses the states of w[i] and b[i].
        Updates the states of z[i] and x[i].

        Args:
            x_in: (n_(i-1), b) numpy array, input of the i-th linear layer
            func: string, either "sigmoid" or "relu" determining the activation
                of the i-th layer
            i: int, layer id

        Returns:
            x_out: (n_i, b) numpy array, output of the i-th linear layer
        """
        #
        # You code here
        #
        b = np.tile(self.b[i], (1, x_in.shape[1])) # (n_i, 1) -> (n_i, b) copy
        self.z[i] = self.w[i] @ x_in + b # (n_i, b) = (n_i, n_(i-1)) @ (n_(i-1), b) + (n_i, b)
        self.x[i] = self.activation_func(func, self.z[i]) # 'sigmoid' or 'relu', (n_i, b)
        return self.z[i]


    def forward(self, x):
        """ Neural network forward propagation. Use ReLU activations in all but the last layer.
        Use the sigmoid function to output class probabilities in the last layer.
        Calls layer_forward in order to update the states of z[i] and x[i] for all layers i.
        Updates the state of x[0].
    
        Args:
            x: (n_0, b) numpy array, input for the forward pass

        Returns:
            predictions: (1, b) numpy array, the network's predictions.
        """
        #
        # You code here
        #
        self.x[0] = x # input, (n_0, b)
        for i in range(1, self.num_layers): # 1, 2, ..., num_layers - 1
            self.layer_forward(self.x[i-1], 'relu', i) # (n_(i-1), b) -> (n_i, b)
        predictions = self.layer_forward(self.x[self.num_layers - 1], 'sigmoid', self.num_layers) # (n_numlayer = 1, b)
        return predictions


    def layer_backward(self, dx_out, func, i):
        """ Backward propagation through the i-th network layer.
        Uses the states of z[i] and x[i-1], as well as w[i] and b[i].
        Updates the states of dw[i] and db[i].

        Args:
            dx_out: (n_i, b) numpy array, partial derivatives dE/dx_i of error with respect to 
                    the output of the activation function in layer i
            func: string, either "sigmoid" or "relu" determining the activation
                of the i-th layer
            i: int, layer id

        Returns:
            dx_in: (n_(i-1), b) numpy array, partial derivatives dE/dx_(i-1) of error
                with respect to the input of layer i
        """
        #
        # You code here
        #
        # dE/dw_i = dE/dz_i * dz_i/dw_i
        #         = self.activation_func_backward(func, dx_out, z_i) @ (x_(i-1))T
        # (n_i, n_i-1) = (n_i, b) @ (b, n_i-1)
        self.dw[i] = self.activation_func_backward(func, dx_out, self.z[i]) @ np.transpose(self.x[i-1])
        # dE/db_i = dE/dz_i * dz_i/db_i
        #         = self.activation_func_backward(func, dx_out, z_i)
        # (n_i, 1) <- (n_i, b) using np.sum()
        self.db[i] = np.sum(self.activation_func_backward(func, dx_out, self.z[i]), axis = 1, keepdims=True)
        # dE/dx_(i-1) = dE/dz_i * dz_i/dx_(i-1)
        #             = dz_i/dx_(i-1) @ dE/dz_i
        #             = (w_i)T @ self.activation_func_backward(func, dx_out, z_i)
        # (n_i-1, b) = (n_i-1, n_i) @ (n_i, b)
        dx_in = np.transpose(self.w[i]) @ self.activation_func_backward(func, dx_out, self.z[i])
        return dx_in
        

    def back_propagation(self, y):
        """ Neural network backward propagation. Use ReLU activations in all but the last layer.
        Use the sigmoid function in the last layer.
        First, 
        Calls layer_backward in order to update the states of dw[i] and db[i] for all layers i.
    
        Args:
            y: (1, b) numpy array, labels needed to back propagate the error

        Returns:
            dx_in: (n_0, b) numpy array, partial derivatives dE/dx_0 of error
                with respect to the network's input
        """
        batch_size = y.shape[1]
        # get predictions from the state dict
        predictions = self.x[self.num_layers]
        # compute the derivative of the mean error regarding the network's output
        d_predictions = - (np.divide(y, predictions) - np.divide(1 - y, 1 - predictions)) / batch_size
        # backward pass through the output layer, updates states of dw and db for the last layer
        dx_in = self.layer_backward(d_predictions, "sigmoid", self.num_layers)
        # iteratively perform backward propagation through the network layers,
        # update states of dw and db for the i-th layer
        for i in reversed(range(1, self.num_layers)):
            dx_in =  self.layer_backward(dx_in, "relu", i)

        return dx_in


    def update_wb(self, lr):
        """ Update the states of w[i] and b[i] for all layers i based on gradient information
        stored in dw[i] and db[i] and the learning rate.

        Args:
            lr: learning rate
        """
        #
        # You code here
        #
        for i in range(1, self.num_layers + 1): # 1, 2, ..., num_layers
            self.w[i] = self.w[i] - lr * self.dw[i]
            self.b[i] = self.b[i] - lr * self.db[i]


    def shuffle_data(self, X, Y):
        """ Shuffles the data arrays X and Y randomly. You can use
        np.random.permutation for this method. Make sure that the label
        belonging X_shuffled[:,i] is shuffled to Y_shuffled[:,i].

        Args:
            X: (n_0, B) numpy array, B feature vectors with N_0-dimensional features
            Y: (1, B) numpy array, labels

        Returns:
            X_shuffled: (n_0, B) numpy array, shuffled version of X
            Y_shuffled: (1, B) numpy array, shuffled version of Y
        """
        #
        # You code here
        #
        X_Y_swapped = np.concatenate((X, Y), axis = 0).swapaxes(0, 1) # (n_0 + 1, B) -> (B, n_0 + 1)
        X_Y_shuffled = np.random.permutation(X_Y_swapped).swapaxes(0, 1) # (B, n_0 + 1) -> (n_0 + 1, B)
        return X_Y_shuffled[:-1], X_Y_shuffled[-1:] # (n_0, B) (1, B)


    def train(self, X, Y, lr, batch_size, num_epochs):
        """ Trains the neural network with stochastic gradient descent by calling
        shuffle_data once per epoch and forward, back_propagation and update_wb
        per iteration. Start a new epoch if the number of remaining data points
        not yet used in the epoch is smaller than the mini batch size.

        Args: 
            X: (n_0, B) numpy array, B feature vectors with N_0-dimensional features
            Y: (1, B) numpy array, labels
            lr: learning rate
            batch_size: mini batch size for SGD
            num_epochs: number of training epochs
        """
        num_examples = X.shape[1]
        it_per_epoch = num_examples // batch_size
        for _ in range(num_epochs):
            X, Y = self.shuffle_data(X, Y)
            for i in range(it_per_epoch):
                # extract mini batches
                x = X[:, i * batch_size : (i+1) * batch_size]
                y = Y[:, i * batch_size : (i+1) * batch_size]
                # perform a forward pass, update states of x and z
                _ = self.forward(x)
                # update states of dw and db by performing a backward pass
                _ = self.back_propagation(y)
                # update states of w and b by a SGD step
                self.update_wb(lr)

