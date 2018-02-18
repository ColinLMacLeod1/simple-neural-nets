import numpy as np

from BackPropNetwork import BackPropNetwork

#Data preparation class
def prep_data(f):
    data = np.loadtxt(f, delimiter = ',')
    train = []
    out = []
    for i in range(len(data)):
        tmp4 = []
        for j in range(11):
            tmp4.append(data[i][j])
        out.append(data[i][11])
        train.append(tmp4)
    return (train, out)



(total, out) = prep_data('wine_data.csv')
norm_out = out/max(out)

# Adding 70% - Train 30% - Test for the low output
train = total[0:1021]
train_out = out[0:1021]
test = total[1022:1458]
test_out = out[1022:1458]
# Adding 70% - Train 30% - Test for the mid output
train.append(total[])
train_out.append(out[])
test.append(total[])
test_out.append(out[])
# Adding 70% - Train 30% - Test for the high output
train.append(total[])
train_out.append(out[])
test.append(total[])
test_out.append(out[])




#
# for i in range(len(norm_out)):
#     if i%4==0:
#         print(i)
#         test.append(total[i])
#         test_out.append(norm_out[i])
#     else:
#         train.append(total[i])
#         train_out.append(norm_out[i])

print(len(test_out), len(train_out))
net = BackPropNetwork(input_size = 11, hidden_size = 12, output_size = 1, learning_rate = 0.5, momentum = 0.1)

net.train(train, train_out, iterations = 20)
net.test(test, test_out)
