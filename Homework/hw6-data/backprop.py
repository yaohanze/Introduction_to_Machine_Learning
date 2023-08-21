import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Gradient descent optimization
# The learning rate is specified by eta
class GDOptimizer(object):
    def __init__(self, eta):
        self.eta = eta

    def initialize(self, layers):
        pass

    # This function performs one gradient descent step
    # layers is a list of dense layers in the network
    # g is a list of gradients going into each layer before the nonlinear activation
    # a is a list of of the activations of each node in the previous layer going
    #
    def update(self, layers, g, a):
        m = a[0].shape[1]
        for layer, curGrad, curA in zip(layers, g, a):
            # TODO: PART E #########################################################################
            # Compute the gradients for layer.W and layer.b using the gradient for the output of the
            # layer curA and the gradient of the output curGrad
            # Use the gradients to update the weight and the bias for the layer
            #
            # Normalize the learning rate by m (defined above), the number of training examples input
            # (in parallel) to the network.
            #
            # It may help to think about how you would calculate the update if we input just one
            # training example at a time; then compute a mean over these individual update values.
            # ######################################################################################
            pass

# Cost function used to compute prediction errors
class QuadraticCost(object):

    # Compute the squared error between the prediction yp and the observation y
    # This method should compute the cost per element such that the output is the
    # same shape as y and yp
    @staticmethod
    def fx(y,yp):
        # TODO: PART B #########################################################################
        # Implement me
        # ######################################################################################
        pass

    # Derivative of the cost function with respect to yp
    @staticmethod
    def dx(y,yp):
        # TODO: PART B #########################################################################
        # Implement me
        # ######################################################################################
        pass

# Sigmoid function fully implemented as an example
class SigmoidActivation(object):
    @staticmethod
    def fx(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def dx(z):
        return SigmoidActivation.fx(z) * (1 - SigmoidActivation.fx(z))

# Hyperbolic tangent function
class TanhActivation(object):

    # Compute tanh for each element in the input z
    @staticmethod
    def fx(z):
        # TODO: PART C #################################################################################
        # Implement me
        # ######################################################################################
        pass

    # Compute the derivative of the tanh function with respect to z
    @staticmethod
    def dx(z):
        # TODO: PART C #########################################################################
        # Implement me
        # ######################################################################################
        pass

# Rectified linear unit
class ReLUActivation(object):
    @staticmethod
    def fx(z):
        # TODO: PART C #########################################################################
        # Implement me
        # ######################################################################################
        pass

    @staticmethod
    def dx(z):
        # TODO: PART C #########################################################################
        # Implement me
        # ######################################################################################
        pass

# Linear activation
class LinearActivation(object):
    @staticmethod
    def fx(z):
        # TODO: PART C #########################################################################
        # Implement me
        # ######################################################################################
        pass

    @staticmethod
    def dx(z):
        # TODO: PART C #########################################################################
        # Implement me
        # ######################################################################################
        pass

# This class represents a single hidden or output layer in the neural network
class DenseLayer(object):

    # numNodes: number of hidden units in the layer
    # activation: the activation function to use in this layer
    def __init__(self, numNodes, activation):
        self.numNodes = numNodes
        self.activation = activation

    def getNumNodes(self):
        return self.numNodes

    # Initialize the weight matrix of this layer based on the size of the matrix W
    def initialize(self, fanIn, scale=1.0):
        s = scale * np.sqrt(6.0 / (self.numNodes + fanIn))
        np.random.seed(0)
        self.W = np.random.normal(0, s,
                                   (self.numNodes,fanIn))
        self.b = np.random.uniform(-1,1,(self.numNodes,1))

    # Apply the activation function of the layer on the input z
    def a(self, z):
        return self.activation.fx(z)

    # Compute the linear part of the layer
    # The input a is an n x k matrix where n is the number of samples
    # and k is the dimension of the previous layer (or the input to the network)
    def z(self, a):
        return self.W.dot(a) + self.b # Note, this is implemented where we assume a is k x n

    # Compute the derivative of the layer's activation function with respect to z
    # where z is the output of the above function.
    # This derivative does not contain the derivative of the matrix multiplication
    # in the layer.  That part is computed below in the model class.
    def dx(self, z):
        return self.activation.dx(z)

    # Update the weights of the layer by adding dW to the weights
    def updateWeights(self, dW):
        self.W = self.W + dW

    # Update the bias of the layer by adding db to the bias
    def updateBias(self, db):
        self.b = self.b + db

# This class handles stacking layers together to form the completed neural network
class Model(object):

    # inputSize: the dimension of the inputs that go into the network
    def __init__(self, inputSize):
        self.layers = []
        self.inputSize = inputSize

    # Add a layer to the end of the network
    def addLayer(self, layer):
        self.layers.append(layer)

    # Get the output size of the layer at the given index
    def getLayerSize(self, index):
        if index >= len(self.layers):
            return self.layers[-1].getNumNodes()
        elif index < 0:
            return self.inputSize
        else:
            return self.layers[index].getNumNodes()

    # Initialize the weights of all of the layers in the network and set the cost
    # function to use for optimization
    def initialize(self, cost, initializeLayers=True):
        self.cost = cost
        if initializeLayers:
            for i in range(0,len(self.layers)):
                if i == len(self.layers) - 1:
                    self.layers[i].initialize(self.getLayerSize(i-1))
                else:
                    self.layers[i].initialize(self.getLayerSize(i-1))

    # Compute the output of the network given some input a
    # The matrix a has shape n x k where n is the number of samples and
    # k is the dimension
    # This function returns
    # yp - the output of the network
    # a - a list of inputs for each layer of the newtork where
    #     a[i] is the input to layer i
    #     (note this does not include the network output!)
    # z - a list of values for each layer after evaluating layer.z(a) but
    #     before evaluating the nonlinear function for the layer
    def evaluate(self, x):
        curA = x.T
        a = [curA]
        z = []
        for layer in self.layers:
            z.append(layer.z(curA))
            curA = layer.a(z[-1])
            a.append(curA)
        yp = a.pop()
        return yp, a, z

    # Compute the output of the network given some input a
    # The matrix a has shape n x k where n is the number of samples and
    # k is the dimension
    def predict(self, a):
        a,_,_ = self.evaluate(a)
        return a.T

    # Computes the gradients at each layer. y is the true labels, yp is the
    # predicted labels, and z is a list of the intermediate values in each
    # layer. Returns the gradients and the forward pass outputs (per layer).
    #
    # In particular, we compute dMSE/dz_i. The reasoning behind this is that
    # in the update function for the optimizer, we do not give it the z values
    # we compute from evaluating the network.
    def compute_grad(self, x, y):
        # Feed forward, computing outputs of each layer and
        # intermediate outputs before the non-linearities
        yp, a, z = self.evaluate(x)

        # d represents (dMSE / da_i) that you derive in part (e);
        #   it is inialized here to be (dMSE / dyp)
        d = self.cost.dx(y.T, yp)
        grad = []

        # Backpropogate the error
        for layer, curZ in zip(reversed(self.layers),reversed(z)):
            # TODO: PART D #########################################################################
            # Compute the gradient of the output of each layer with respect to the error
            # grad[i] should correspond with the gradient of the output of layer i
            #   before the activation is applied (dMSE / dz_i); be sure values are stored
            #   in the correct ordering!
            # ######################################################################################
            pass

        return grad, a

    # Computes the gradients at each layer. y is the true labels, yp is the
    # predicted labels, and z is a list of the intermediate values in each
    # layer. Uses numerical derivatives to solve rather than symbolic derivatives.
    # Returns the gradients and the forward pass outputs (per layer).
    #
    # In particular, we compute dMSE/dz_i. The reasoning behind this is that
    # in the update function for the optimizer, we do not give it the z values
    # we compute from evaluating the network.
    def numerical_grad(self, x, y, delta=1e-4):

        # computes the loss function output when starting from the ith layer
        # and inputting z_i
        def compute_cost_from_layer(layer_i, z_i):
            cost = self.layers[layer_i].a(z_i)
            for layer in self.layers[layer_i+1:]:
                cost = layer.a(layer.z(cost))
            return self.cost.fx(y.T, cost)

        # numerically computes the gradient of the error with respect to z_i
        def compute_grad_from_layer(layer_i, inp):
            mask = np.zeros(self.layers[layer_i].b.shape)
            grad_z = []
            # iterate to compute gradient of each variable in z_i, one at a time
            for i in range(mask.shape[0]):
                mask[i] = 1
                delta_p_output = compute_cost_from_layer(layer_i, inp+mask*delta)
                delta_n_output = compute_cost_from_layer(layer_i, inp-mask*delta)
                grad_z.append((delta_p_output - delta_n_output) / (2 * delta))
                mask[i] = 0;

            return np.vstack(grad_z)

        _, a, _ = self.evaluate(x)

        grad = []
        i = 0
        curA = x.T
        for layer in self.layers:
            curA = layer.z(curA)
            grad.append(compute_grad_from_layer(i, curA))
            curA = layer.a(curA)
            i += 1


        return grad, a

    # Train the network given the inputs x and the corresponding observations y
    # The network should be trained for numEpochs iterations using the supplied
    # optimizer
    def train(self, x, y, numEpochs, optimizer):

        # Initialize some stuff
        n = x.shape[0]
        x = x.copy()
        y = y.copy()
        hist = []
        optimizer.initialize(self.layers)

        # Run for the specified number of epochs
        for epoch in range(0,numEpochs):
 
            # Compute the gradients
            grad, a = self.compute_grad(x, y)

            # Update the network weights
            optimizer.update(self.layers, grad, a)

            # Compute the error at the end of the epoch
            yh = self.predict(x)
            C = self.cost.fx(y, yh)
            C = np.mean(C)
            hist.append(C)
        return hist

if __name__ == '__main__':
    # switch these statements to True to run the code for the corresponding parts
    DEBUG_MODEL = False
    # PART E
    SGD = False
    # Part F
    DIFF_SIZES = False


    # Generate the training set
    np.random.seed(9001)
    x=np.random.uniform(-np.pi,np.pi,(1000,1))
    y=np.sin(x)
    xLin=np.linspace(-np.pi,np.pi,250).reshape((-1,1))
    yHats = {}

    activations = dict(ReLU=ReLUActivation,
                       tanh=TanhActivation,
                       linear=LinearActivation)
    lr = dict(ReLU=0.02,tanh=0.02,linear=0.005)
    names = ['ReLU','linear','tanh']

    if DEBUG_MODEL:
        print('Debugging gradients..')
        # Build the model
        activation = activations["ReLU"]
        model = Model(x.shape[1])
        model.addLayer(DenseLayer(10,activation()))
        model.addLayer(DenseLayer(10,activation()))
        model.addLayer(DenseLayer(1,LinearActivation()))
        model.initialize(QuadraticCost())

        grad, _ = model.compute_grad(x, y)
        n_grad, _ = model.numerical_grad(x, y)
        for i in range(len(grad)):
            print('squared difference of layer %d:' % i, np.linalg.norm(grad[i] - n_grad[i]))
    
     #### PART E ####
    if SGD:
        print('\n----------------------------------------\n')
        print('Using SGD')
        batchSizes = [1, 128, x.shape[0]]
        for key in names:
            for batchSize in batchSizes:
                # Build the model
                activation = activations[key]
                model = Model(x.shape[1])
                model.addLayer(DenseLayer(4,activation()))
                model.addLayer(DenseLayer(1,LinearActivation()))
                model.initialize(QuadraticCost())
                # Train the model and display the results
                epochs = 1  # Make sure that the same number of gradients steps are taken
                hist = model.trainBatch(x,y,batchSize,epochs,GDOptimizer(eta=lr[key]))
                y_hat_train = model.predict(x)
                y_pred_train = np.sign(y_hat_train)

                y_hat_test = model.predict(X_test)
                y_pred_test = np.sign(y_hat_test)
                error_train = np.mean(np.square(y_hat_train - y))/2
                error_test = np.mean(np.square(y_hat_test - y_test))/2

                print(y_pred_test.ravel())
                print(y_test.ravel())


                print(key+'('+str(batchSize)+') train MSE: '+str(error_train))
                print(key+'('+str(batchSize)+') test MSE: '+str(error_test))
                print("train accuracy", (y_pred_train.ravel() == y_train).sum()/len(y_train))
                print("test accuracy", (y_pred_test.ravel() == y_test).sum()/len(y_test))

    # Train with different sized networks
    #### PART F ####
    if DIFF_SIZES:
        print('\n----------------------------------------\n')
        print('Training with various sized network')
        names = ['ReLU', 'tanh']
        sizes = [5,10,25,50]
        widths = [1,2,3]
        errors = {}
        y = np.sin(x)
        for key in names:
            error = []
            for width in widths:
                for size in sizes:
                    activation = activations[key]
                    model = Model(x.shape[1])
                    for _ in range(width):
                        model.addLayer(DenseLayer(size,activation()))
                    model.addLayer(DenseLayer(1,LinearActivation()))
                    model.initialize(QuadraticCost())
                    hist = model.train(x,y,500,GDOptimizer(eta=lr[key]))
                    yHat = model.predict(x)
                    yHats[key] = model.predict(xLin)
                    e = np.mean(np.square(yHat - y))/2
                    error.append(e)
            errors[key] = np.asarray(error).reshape((len(widths),len(sizes)))

        # Print the results
        for key in names:
            error = errors[key]
            print(key+' MSE Error')
            header = '{:^8}'
            for _ in range(len(sizes)):
                header += ' {:^8}'
            headerText = ['Layers'] + [str(s)+' nodes' for s in sizes]
            print(header.format(*headerText))
            for width,row in zip(widths,error):
                text = '{:>8}'
                for _ in range(len(row)):
                    text += ' {:<8}'
                rowText = [str(width)] + ['{0:.5f}'.format(r) for r in row]
                print(text.format(*rowText))
