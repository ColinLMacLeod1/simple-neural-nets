import math
import random
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dv_sigmoid(y):
    return y*(1.0-y)

class BackPropNetwork(object):
    def __init__(self, input_size, hidden_size, output_size, momentum, learning_rate):
        self.input_size = input_size + 1
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum

        #Initialize weight matrices for both input-hidden and hidden-output
        self.weights_input = np.random.normal(loc = 0, scale = range(input_size), size = (self.input_size, self.hidden_size))
        self.weights_output = np.random.normal(loc = 0, scale = range(hidden_size), size = (self.hidden_size, self.output_size))
        #Input values passed in at each iteration
        self.input_values = [0] * self.input_size
        #Hidden values computed with each iteration
        self.hidden_values = [0] * self.hidden_size
        #Output values computed with each iteration
        self.output_values = [0] * self.output_size

        self.change_in = np.zeros((self.input_size, self.hidden_size))
        self.change_out = np.zeros((self.hidden_size, self.output_size))

    #Feedforward network which accepts individual input vectors
    def feedForward(self, x_vector):
        #Set input values as x_vector
        for i in range(self.input_size - 1):
            self.input_values[i] = x_vector[i]
        #Next, we compute the values at each hidden node for our input values.
        for i in range(self.hidden_size):
            sum = 0
            for j in range(self.input_size):
                sum += self.input_values[i] * self.weights_input[i][j]
            self.hidden_values[i] = sigmoid(sum)
        #Next, use our hidden node values and current weights to compute output values
        for i in range(self.output_size):
            sum = 0
            for j in range(self.hidden_size):
                sum += self.hidden_values[i] * self.weights_output[j][i]
            self.output_values[i] = sigmoid(sum)
        return self.output_values[:]

    #Backpropagation functionality to be executed after each feedforward iteration
    def backPropagate(self, actual_output):
        #Initialize an output vector for this iteration, which will store the computed gamma based on output error and sigmoid derivative
        output_vector = [0] * self.output_size
        for i in range(self.output_size):
            e = -(actual_output[i] - self.output_values[k])
            output_vector[i] = dv_sigmoid(self.output_values[i])*e

        hidden_vector = [0] * self.hidden_size
        for i in range(self.hidden_size):
            e = 0
            for j in range(self.output_size):
                e += output_vector[j] * self.weights_output[j][i]
            hidden_vector[i] = dv_sigmoid(self.hidden_values[i])*e

        #Next, we use our computed gamma values to update the weight matrices for both hidden-output
        #and input-hidden
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                #Compute the product of gamme and each hidden node
                change = output_vector[j] * self.hidden_values[i]
                #Update weights vector for hidden-output
                self.weights_output[i][j] -= self.learning_rate * change + self.change_out[i][j] * self.momentum
                self.change_out[i][j] = change

        for i in range(self.input_size):
            for j in range(self.hidden_size):
                #Compute the product of gamme and each input value
                change = hidden_vector[j] * self.input_values[i]
                #Update weights vector for input-hidden
                self.weights_input[i][j] -= self.learning_rate * change + self.change_in[i][j] * self.momentum
                self.change_in[i][j] = change

        #compute root mean square error
        error = np.zeros(len(actual_output))
        for i in range(len(actual_output)):
            error[i] = 0.5 * (actual_output[i] - self.output_values[i]) ** 2
        mean = np.mean(error)
        e = math.sqrt(mean)
        return e

    # def test(self, inputData, actual_ouput):
    #     f = open('assignment2data.txt', 'w')
    #     mistakes = 0
    #     for i in range(len(inputData)):
    #         output = self.feedForward(inputData[i])

    def train(self, inputData, actual_output, iterations):
        for i in range(iterations):
            error = 0
            for j in range(len(inputData)):
                self.feedForward(input[j])
                error += self.backPropagate(actual_ouput[j])
            print "RMSE: {0}".format(error)
