import math
import random
import numpy as np

def sigmoid(x):
    return math.tanh(x)

def dv_sigmoid(y):
    return 1 - y**2

class BackPropNetwork(object):
    def __init__(self, input_size, hidden_size, output_size, momentum, learning_rate):
        self.input_size = input_size + 1
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum

        #Initialize weight matrices for both input-hidden and hidden-output
        self.weights_input = np.random.normal(loc = 0, scale = 1, size = (self.input_size, self.hidden_size))
        self.weights_output = np.random.normal(loc = 0, scale = 1, size = (self.hidden_size, self.output_size))
        #Input values passed in at each iteration
        self.input_values = [0] * self.input_size
        #Hidden values computed with each iteration
        self.hidden_values = [0] * self.hidden_size
        #Output values computed with each iteration
        #self.output_values = [0] * self.output_size
        self.output_value = 0

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
                sum += self.hidden_values[j] * self.weights_output[j][i]
            self.output_value = sigmoid(sum)
        return self.output_value

    #Backpropagation functionality to be executed after each feedforward iteration
    def backPropagate(self, actual_output):
        #Initialize an output vector for this iteration, which will store the computed gamma based on output error and sigmoid derivative
        output_vector = [0] * self.output_size
        for i in range(self.output_size):
            e = (actual_output - self.output_value)
            output_vector[i] = dv_sigmoid(self.output_value)*e

        hidden_vector = [0] * self.hidden_size
        for i in range(self.hidden_size):
            e = 0
            for j in range(self.output_size):
                e += output_vector[j] * self.weights_output[i][j]
            hidden_vector[i] = dv_sigmoid(self.hidden_values[i])*e

        #Next, we use our computed gamma values to update the weight matrices for both hidden-output
        #and input-hidden
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                #Compute the product of gamme and each hidden node
                change = output_vector[j] * self.hidden_values[i]
                #Update weights vector for hidden-output
                self.weights_output[i][j] += self.learning_rate * change + self.change_out[i][j] * self.momentum
                self.change_out[i][j] = change

        for i in range(self.input_size):
            for j in range(self.hidden_size):
                #Compute the product of gamme and each input value
                change = hidden_vector[j] * self.input_values[i]
                #Update weights vector for input-hidden
                self.weights_input[i][j] += self.learning_rate * change + self.change_in[i][j] * self.momentum
                self.change_in[i][j] = change

        #compute root mean square error
        #error = np.zeros(len(actual_output))
        #for i in range(len(actual_output)):
        error = 0.0
        error = 0.5 * (actual_output - self.output_value) ** 2
        return error

    #interprets final output vector
    #def ff_return(self, a):
 #       new_a = 0
#        if (a > 0.5):
#            new_a = 1
#        return new_a

    def test(self, inputData, actualNum):
        f = open('test-results.txt', 'w')
        incorrect = len(inputData)
        output = [0] * len(inputData)
        temp_a = 0
        curr_out = 0
        for i in range(len(inputData)):
            #use existing feedForward network to predict the letter
            print(inputData[i], '->', self.feedForward(inputData[i]))
            # output[i] = self.feedForward(inputData[i])
        #     temp_a = output[i]
        #     if (temp_a > 0.5):
        #         curr_out = 1
        #     else:
        #         curr_out = 0
        #     #keep track of the number of outputs guessed wrong
        #     if (curr_out == actualNum[i]):
        #         incorrect -= 1
        #         #write out a subset of the data
        #     if i % 15 ==0:
        #         f.write('Predicted: {0} ---> Actual: {1}\n'.format(actualNum[i], curr_out))
        # f.write('\n\nTotal incorrect on test sample: {0}'.format(float(incorrect)/float(len(inputData))))
        # f.close()
        # print("Incorrect: ", incorrect)
        # return

    def train(self, inputData, actual_output, iterations):
        for i in range(iterations):
            error = 0.0
            for j in range(len(inputData)):
                self.feedForward(inputData[j])
                error += self.backPropagate(actual_output[j])
            if (i%100 == 0):
                print('error %-.5f' % error)
            # rmse = np.mean(error)
            # print("RMSE: {0}".format(rmse))
