inputs = [0 0;0 1; 1 0; 1 1];
desOut = [-1 -1 -1 1];
out = [0 0 0 0];
w = [rand(1) rand(1) rand(1)];
f = @(w0,w1,w2,x1,x2) w0+w1*x1+w2*x2;
c = 0.5;
correct = 0;
err = 0;
loweindex = 0;
dw = [0 0 0];
while ~isequal(correct,[1 1 1 1])
    w = w + dw;
    figure(1)
    plot([0 -w(2)/w(3)],[-w(1)/w(3) 0])
    hold on
    for i = 1:length(desOut)
        plot(inputs(i,1), inputs(i,2), 'or')
        x = f(w(1),w(2),w(3),inputs(i,1),inputs(i,2));
        if x<w(1)
            out(i) = -1;
        else
            out(i) = 1;
        end
        if desOut(i) ~= out(i)
            err = out(i) - desOut(i);
        end
        
    end
    index = 1;
    dw = c.*[1 inputs(index,1) inputs(index,2)]*err; 
    hold off
    
    
end

