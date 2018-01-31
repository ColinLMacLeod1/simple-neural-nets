%% CMPE 452 - Assignment 1
[x,t] = cancer_dataset; % x is the input and t is the target
% whos
setdemorandstream(672880951) % To reduce the randomness of the initial weights
net = patternnet(5); % 5 layer net
% view(net) % Display a graphic of the net
[net,tr] = train(net,x,t); % Train the set and automatically dividing into training validation and testing

testX = x(:,tr.testInd);
testT = t(:,tr.testInd);
testY = net(x);
testClasses = testY > 0.5;
accuracy=sum(testClasses(1,:)==t(1,:))/699
%confusion()