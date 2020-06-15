import numpy as np
import matplotlib.pyplot as plt
import pickle
from numpy import genfromtxt


# Class to train MLP model
class MLP(object):

    def __init__(self, layers):
        self.layers = layers
        self.epochs = 55  # number of epochs used to train the entire data-set
        self.weights = []
        self.bias = []
        self.trainError = []  # To store the training error after each epoch
        self.valError = []  # To store the validation error after each epoch
        self.step_size = 0.001  # Learning Rate used
        self.tol = 0.0001  # Tolerance chosen for loss function

        # Initialize weights and bias for the second layer to be random between -1 and 1 and for the last layer to be 0.
        for i in range(len(self.layers) - 1):
            if i == 0:
                self.weights.append(np.random.uniform(-1, 1, (layers[i + 1], layers[i])))
                self.bias.append(np.random.uniform(-1, 1, (layers[i + 1], 1)))
            else:
                self.weights.append(np.zeros((layers[i + 1], layers[i])))
                self.bias.append(np.zeros((layers[i + 1], 1)))

    # Sigmoid Activation Function for the hidden layer perceptrons
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    # Sigmoid Derivative for back propagation for the hidden layer
    def sigmoidDerivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    # Softmax Activation Function for the last layer perceptrons
    def softmax(self, A):
        expA = np.exp(A)
        return expA / expA.sum()

    # Cross-entropy Error as loss function
    def costFunctionCrossentropy(self, y, a_lastLayer):
        return -1 * np.sum(y * np.log(a_lastLayer))

    # Derivative of Cross-entropy loss function and softmax function
    def crossEntropyDerivative(self, y, a_lastLayer):
        return a_lastLayer - y

    # Feed-forward Propagation to calculate all the activations of all the layers for each training example
    def feedForward(self, x, activations, z):

        x = x.reshape(len(x), 1)
        activations.append(x)
        # Iterate over layers to find the activations
        for layer in range(2):
            if layer == 0:  # For Sigmoid
                z1 = np.dot(self.weights[layer], activations[layer]) + self.bias[layer]
                z.append(z1)
                activations.append(self.sigmoid(z1))
            else:  # For Softmax
                z1 = np.dot(self.weights[layer], activations[layer]) + self.bias[layer]
                z.append(z1)
                activations.append(self.softmax(z1))

        return activations, z

    # Back-Propagation function to update weights according to the gradients.
    def backPropagate(self, y, activations, z):

        grad_w = []  # To store the change in weights dor each training example
        grad_b = []  # To store the change in bias dor each training example

        # Appending vectors of same shape as weights and bias of each layer and initializing them with 0.
        for i in range(len(self.layers) - 1):
            grad_w.append(np.zeros((self.layers[i + 1], self.layers[i])))
            grad_b.append(np.zeros((self.layers[i + 1], 1)))

        # Last Layer Delta
        delta_last_layer = self.crossEntropyDerivative(y, activations[-1])
        # gradient of weights for the last layer
        grad_w[-1] = np.dot(delta_last_layer, activations[-2].T)
        # gradient of bias for the last layer
        grad_b[-1] = delta_last_layer

        # Second Layer Delta
        delta_second_layer = np.dot(self.weights[-1].T, delta_last_layer) * self.sigmoidDerivative(z[-2])
        # gradient of weights for the hidden layer
        grad_w[-2] = delta_second_layer.dot(activations[-3].T)
        # gradient of bias for the hidden layer
        grad_b[-2] = delta_second_layer

        return grad_w, grad_b

    # Function to evaluate Validation Error after each epoch to determine Overfitting or Underfitting of the model.
    def test_function(self, X, y):
        error = 0
        for i in range(len(X)):
            activations = []
            activations, _ = self.feedForward(X[i], activations, [])
            # Calculate validation error for each validating example
            error = error + self.costFunctionCrossentropy(y[i].reshape(len(y[i]), 1), activations[-1])

        return error / len(X)  # Averaging the error

    # Training the network
    def train(self, X_train, y_train, X_val, y_val):

        # Loop over number of epochs
        for epoch in range(self.epochs):
            # Shuffle data to avoid Over-fitting
            shuffler = np.random.permutation(len(X_train))
            X_train = X_train[shuffler]
            y_train = y_train[shuffler]
            error = []  # Store error for each training sample

            # Loop for iterating over training data-set.
            for i in range(len(X_train)):
                # To store activations and z values(values f=before applying any activation function)
                activations = []
                z = []
                # Estimate activations using feed-forward function
                activations, z = self.feedForward(X_train[i], activations, z)
                # Estimate training error for each sample using Cross-entropy cost function
                sample_error = self.costFunctionCrossentropy(y_train[i].reshape(len(y_train[i]), 1), activations[-1])
                # Appending the error to the error-list for each epoch to calculate the average error of an epoch
                error.append(sample_error)
                # Estimate the change in weights and bias due to each training sample using back-propagation function
                grad_w, grad_b = self.backPropagate(y_train[i].reshape(len(y_train[i]), 1), activations, z)
                # Update weights and bias according to each training sample
                for k in range(2):
                    self.weights[k] = self.weights[k] - (self.step_size * grad_w[k])
                    self.bias[k] = self.bias[k] - (self.step_size * grad_b[k])

            # Completion of an Epoch!!
            # Estimate Training and Validation Error for each epoch:
            val_error_epoch = self.test_function(X_val, y_val)
            train_error_epoch = np.sum(error) / len(error)
            self.trainError.append(train_error_epoch)
            self.valError.append(val_error_epoch)
            print("EPOCH ", epoch + 1, "/", self.epochs, "---> Training Error: ", train_error_epoch,
                  " Training Accuracy: ", 1 - train_error_epoch, " Validation Error: ", val_error_epoch,  "Validation "
                                                                                                          "Accuracy: "
                                                                                                          "",
                  1-val_error_epoch)

            # Stopping Condition for the loss
            if self.trainError[- 1] == self.tol:
                print("Stopping Criteria reached as training loss is less than the tolerance.")
                break


# Import Data using numpy
def importData():
    X = genfromtxt('train_data.csv', delimiter=',')
    y = genfromtxt('train_labels.csv', delimiter=',')
    return X, y


# Split data-set into training and testing sets(80:20)
def splitData(X, y):
    n = np.round(0.8 * len(X)).astype(int)
    trainX, valX = np.split(X, [n])
    trainY, valY = np.split(y, [n])
    return trainX, trainY, valX, valY


# main function
def main():

    # Import Data
    X, y = importData()
    # Split data-set into training and validation sets
    X_train, y_train, X_val, y_val = splitData(X, y)
    # Creating an object of MLP model and passing the number of layers and hidden nodes in each layer
    mlpObject = MLP(layers=[len(X_train[0]), 150, 4])
    # Training the model
    mlpObject.train(X_train, y_train, X_val, y_val)
    # Store the Final weights and Bias estimated via training
    final_weights = mlpObject.weights
    final_bias = mlpObject.bias

    # Plot training and validation metrics
    # Training & Validation Errors
    plt.plot(mlpObject.trainError, label='Training Error')
    plt.plot(mlpObject.valError, label='Validation Error')
    plt.legend()
    plt.show()

    # Training & Validation Accuracy
    trainAcc = []
    valAcc = []

    for i in range(len(mlpObject.trainError)):
        trainAcc.append(1 - mlpObject.trainError[i])
    for i in range(len(mlpObject.valError)):
        valAcc.append(1 - mlpObject.valError[i])
    plt.plot(trainAcc, label='Training Accuracy')
    plt.plot(valAcc, label='Validation Accuracy')
    plt.legend()
    plt.show()

    # Store the weights and bias in pickle file
    model_output = [final_weights, final_bias]
    with open('mlpModel.pkl', 'wb') as f:
        pickle.dump(model_output, f)


if __name__ == '__main__':
    main()
