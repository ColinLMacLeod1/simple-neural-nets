import math
import random
import numpy as np

def sigmoid(s):
    return 1/(1+np.exp(-s))

def dsigmoid(s):
    return s * (1 - s)

class Network(object):
    def __init__(self, lRate, momentum, inSize, hidSize, outSize):
        self.lRate = lRate
        self.momentum = momentum
        self.inSize = inSize
        self.hidSize = hidSize
        self.outSize = outSize
        self.inWeights = np.random.randn(self.inSize, self.hidSize)
        self.outWeights = np.random.randn(self.hidSize, self.outSize)
        self.inVals = [0] * self.inSize
        self.hidVals = [0] * self.hidSize
        self.outVal = 0
        self.deltaIn = np.zeros((self.inSize, self.hidSize))
        self.deltaOut = np.zeros((self.hidSize, self.outSize))

    def backP(self, expected):
        outVec = [0] * self.outSize
        for i in range(self.outSize):
            e = (expected - self.outVal)
            outVec[i] = dsigmoid(self.outVal)*e
        hidVec = [0] * self.hidSize
        for i in range(self.hidSize):
            e = 0
            for j in range(self.outSize):
                e += outVec[j] * self.outWeights[i][j]
            hidVec[i] = dsigmoid(self.hidVals[i])*e
        for i in range(self.hidSize):
            for j in range(self.outSize):
                delta = outVec[j] * self.hidVals[i]
                self.outWeights[i][j] += self.lRate * delta + self.deltaOut[i][j] * self.momentum
                self.deltaOut[i][j] = delta
        for i in range(self.inSize):
            for j in range(self.hidSize):
                delta = hidVec[j] * self.inVals[i]
                self.inWeights[i][j] += self.lRate * delta + self.deltaIn[i][j] * self.momentum
                self.deltaIn[i][j] = delta
        error = 0.0
        error = 0.5 * (expected - self.outVal) ** 2
        return error

    def forward(self, x_vector):
        for i in range(self.inSize - 1):
            self.inVals[i] = x_vector[i]
        for i in range(self.hidSize):
            sum = 0
            for j in range(self.inSize):
                sum += self.inVals[i] * self.inWeights[i][j]
            self.hidVals[i] = sigmoid(sum)
        for i in range(self.outSize):
            sum = 0
            for j in range(self.hidSize):
                sum += self.hidVals[j] * self.outWeights[j][i]
            self.outVal = sigmoid(sum)
        return self.outVal

    def train(self, inputData, expected, iterations):
        for i in range(iterations):
            error = 0.0
            for j in range(len(inputData)):
                x = self.forward(inputData[j])
                error += self.backP(expected[j])
            rmse = np.mean(error)
            print("RMS Error: {0}".format(rmse))

    def test(self, inputData, actualNum):
        f = open('results.txt', 'w')
        incorrect = 0
        out = [0] * len(inputData)
        temp = 0
        tempOut = 0
        for i in range(len(inputData)):
            out[i] = self.forward(inputData[i])
            temp = out[i]
            if (temp < 0.625):
                tempOut = 0.625
            elif (temp < 0.875):
                tempOut = 0.875
            else:
                tempOut = 1
            if (tempOut != actualNum[i]):
                incorrect +=1
            f.write('Expected: {0}, Output: {1}\n'.format(actualNum[i], tempOut))
        f.write('\n\nIncorrect Precentage: {0}'.format(float(incorrect)/float(len(inputData))))
        f.write('\n\nFinal Weights: \n-Input: {0}\n-Output: {1}'.format(self.inWeights, self.outWeights))
        f.close()
        return
