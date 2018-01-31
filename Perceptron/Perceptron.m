%% CMPE 452 Assignment 1 - Perceptron
% Colin MacLeod 101051666
% There is something wrong with my training and it blow up making all of
% the outputs of a neuraon all 1 or all 0. It is also strange that the 3rd
% neuron tends to work pretty well but the first two don't. I will continue
% to try to figure it out but I need to submit.
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

w1 = [rand() rand() rand() rand() rand()]*2-1;
% weight vector for each output node
w2 = [rand() rand() rand() rand() rand()]*2-1;  % [w0 w1 w2 w3 w4]
w3 = [rand() rand() rand() rand() rand()]*2-1;
weights = [w1;w2;w3];
c = 0.1; % Learning Rate


trainOut = zeros(length(trainExp),3); % Training outputs
testOut = zeros(length(testExp),3); % Testing Outputs
trainCorr = zeros(length(trainExp),3); % Correct answers in training
testCorr = zeros(length(testExp),3); % Correst answers in testing

f = @(w,x) w(1)+w(2)*x(1)+w(3)*x(2)+w(4)*x(3)+w(5)*x(4);
correct = [0 0 0];
err = zeros(length(trainOut),3);
iterations = 0;
%% Training
while correct(1)<80 || correct(2)<80 || correct(3)<81 % 
    iterations = iterations+1;
    for i=1:length(trainExp)
        for j=1:3
            if f(weights(j,:),trainRawIn(i,:))>0
                trainOut(i,j) = 1;
            else trainOut(i,j) = 0;
            end
            if trainOut(i,j) == trainExp(i,j)
                trainCorr(i,j)=1;
                err(i,j) = inf;
            else
                err(i,j) = f(weights(j,:),trainRawIn(i,:));
                trainCorr(i,j)=0;
            end
        end
    end
    [minErr,index] = min(abs(err));
    correct = sum(trainCorr)
    for z = 1:3
        weights(z,:) = weights(z,:) - c*err(index(z),z)*[1,trainRawIn(index(z),:)];
    end
end

%% Testing 
for i=1:length(testExp)
    for j=1:3
        if f(weights(j,:),testRawIn(i,:))>0
            testOut(i,j) = 1;
        else testOut(i,j) = 0;
        end
        if testOut(i,j) == testExp(i,j)
            testCorr(i,j)=1;
            err(i,j) = inf;
        else
            err(i,j) = f(weights(j,:),testRawIn(i,:));
            testCorr(i,j)=0;
        end
    end
end
PetalWidth = testRawIn(:,1);
PetalLength = testRawIn(:,2);
SepalWidth = testRawIn(:,3);
SepalLength = testRawIn(:,4);
SetosaOut = testOut(:,1);
VersicolorOut = testOut(:,2);
VirginiaOut = testOut(:,3);

accuracy = sum(testCorr)/30*100;
FinalTable = table(PetalWidth,PetalLength,SepalWidth,SepalLength,SetosaOut,VersicolorOut,VirginiaOut);
Weights = table(weights);
filename = 'CMPE452Assn1.xlsx';
writetable(FinalTable,filename,'Sheet',1)
writetable(Weights,filename,'Sheet',2)
fprintf('The program iterated %d times', iterations)


 



