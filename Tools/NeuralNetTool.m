%% CMPE 452 Assignment 1 - Tool
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
x = [trainRawIn; testRawIn]';
t = [trainExp; testExp]';
setdemorandstream(672880951) % reduce the randomness of the initial weights
net = patternnet(5); % 5 layer net
view(net) % Display  the net
[net,tr] = train(net,x,t); % Train the set and automatically dividing into training validation and testing

testX = x(:,tr.testInd);
testT = t(:,tr.testInd);
testY = net(x);
testClasses = testY > 0.5;
accuracy=sum(testClasses(1,:)==t(1,:))/150
%confusion()