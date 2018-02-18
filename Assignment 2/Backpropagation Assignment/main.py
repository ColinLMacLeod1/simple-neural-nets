import numpy as np
from Network import Network

data = np.loadtxt('wine_data.csv', delimiter = ',')
total = []
out = []
for i in range(len(data)):
    inVector = []
    for j in range(11):
        inVector.append(data[i][j])
    out.append(data[i][11])
    total.append(inVector)

out = out/max(out)
train = []
trainOut = []
test = []
testOut = []
for i in range(len(out)):
    if i%4==0:
        test.append(total[i])
        testOut.append(out[i])
    else:
        train.append(total[i])
        trainOut.append(out[i])

print(testOut)
print(len(testOut), len(trainOut))
NN = Network(lRate = 0.1, momentum = 0.1, inSize = 11+1, hidSize = 12, outSize = 1) #The 1 was added to input size for bias

NN.train(train, trainOut, iterations = 50)
NN.test(test, testOut)
