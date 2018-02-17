#File to prepare data and run training/testing
import numpy as np

from BackPropNetwork import BackPropNetwork

#Data preparation class
def prep_data(f):
    data = np.loadtxt(f, delimiter = ',')
    low_dat = []
    med_dat = []
    high_dat = []
    low_out = []
    med_out = []
    high_out = []
    train = []
    for i in range(len(data)):
        tmp1 = []
        tmp2 = []
        tmp3 = []
        tmp4 = []
        for j in range(11):
            tmp1.append(data[i][j])
            tmp2.append(data[i][j])
            tmp3.append(data[i][j])
            tmp4.append(data[i][j])
        low_dat.append(tmp1)
        med_dat.append(tmp2)
        high_dat.append(tmp3)
        train.append(tmp4)
        if (data[i][11] == 5):
            low_dat[i].append(1)
            med_dat[i].append(0)
            high_dat[i].append(0)
            low_out.append(1)
            med_out.append(0)
            high_out.append(0)
        elif (data[i][11] == 7):
            low_dat[i].append(0)
            med_dat[i].append(1)
            high_dat[i].append(0)
            low_out.append(0)
            med_out.append(1)
            high_out.append(0)
        else:
            low_dat[i].append(0)
            med_dat[i].append(0)
            high_dat[i].append(1)
            low_out.append(0)
            med_out.append(0)
            high_out.append(1)
    #delete predicted values from matrix (last column)
    #train = np.delete(train, (11), axis=1)
    return (low_out, med_out, high_out, data, train, low_dat, med_dat, high_dat)



(low_out, med_out, high_out, data, train, low_dat, med_dat, high_dat) = prep_data('wine_data.csv')
#See README for explaination of value choices
five = BackPropNetwork(input_size = 11, hidden_size = 12, output_size = 1, learning_rate = 0.5, momentum = 0.1)
seven = BackPropNetwork(input_size = 11, hidden_size = 12, output_size = 1, learning_rate = 0.5, momentum = 0.1)
eight = BackPropNetwork(input_size = 11, hidden_size = 12, output_size = 1, learning_rate = 0.5, momentum = 0.1)

print("About to train")
five.train(train, low_out, iterations = 1000)
five.test(train, low_out)
