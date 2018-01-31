%% CMPE 452 Assignment 1 - Perceptron
% Colin MacLeod 101051666
%% Import Data
%[trainRawIn, trainRawOut] = xlsread('data/train.xlsx');
%[testRawIn, testRawOut] = xlsread('data/test.xlsx');
trainExp = zeros(length(trainRawOut),3); % Initializing the expected output array
testExp = zeros(length(testRawOut),3); % Initializing the expected output array
%% Create all necesaccy vectors
for i=1:length(trainExp) %Training output vector
    if strcmp(trainRawOut(i),'Iris-setosa')
        trainExp(i,1) = 1;
    elseif strcmp(trainRawOut(i),'Iris-versicolor')
        trainExp(i,2) = 1;
    elseif strcmp(trainRawOut(i),'Iris-virginica')
        trainExp(i,3) = 1;
    end   
end
for i=1:length(testExp) % test output vector
    if strcmp(testRawOut(i),'Iris-setosa')
        testExp(i,1) = 1;
    elseif strcmp(trainRawOut(i),'Iris-versicolor')
        testExp(i,2) = 1;
    elseif strcmp(trainRawOut(i),'Iris-virginica')
        testExp(i,3) = 1;
    end   
end

w1 = [rand() rand() rand() rand() rand()]*2-1 % weight vector for each output node
w2 = [rand() rand() rand() rand() rand()]*2-1  % [w0 w1 w2 w3 w4]
w3 = [rand() rand() rand() rand() rand()]*2-1 
c = 0.1; % Learning Rate


trainOut = zeros(length(trainExp),3); % Training outputs
testOut = zeros(length(testExp),3); % Testing Outputs
trainCorr = zeros(length(trainExp),3); % Correct answers in training
testCorr = zeros(length(testExp),3); % Correst answers in testing

f = @(w,x) w(1)+w(2)*x(1)+w(3)*x(2)+w(4)*x(3)+w(5)*x(4);
x=59;
%% Training
while x < 60 % 
    for i=1:length(trainExp)
        for j=1:3
            if f(w1,trainRawIn(i,:))>0
                trainOut(i,j) = 1;
            else trainOut(i,j) = 0;
            end
            if trainOut(i,j) == trainExp(i,j)
                trainCorr(i,j)=1;
            else
                trainCorr(i,j)=0;
            end
        end
    end
    sum(trainCorr)
    x=61;
end




 



