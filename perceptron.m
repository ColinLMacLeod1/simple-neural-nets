inputs = [0 0;0 1; 1 0; 1 1];
desOut = [0 0 0 1];
out = [0 0 0 0];
w = [rand(1) rand(1) rand(1)];
LSep = @(w0,w1,w2,x1) (-w1/w2)*x1 - w0/w1;
lRate = 1;
correct = [0 0 0 0];
err = [0 0 0 0];
dw = [0 0 0];
while ~isequal(correct,[1 1 1 1])
    figure(1)
    plot(0:2, LSep(w(1),w(2),w(3),0:2))
    hold on
    for i = 1:length(desOut)
        
       
        if desOut(i) == 1
            plot(inputs(i,1),inputs(i,2),'bo')
        else
            plot(inputs(i,1),inputs(i,2),'ro')
        end
        out(i) = LSep(w(1),w(2),w(3),inputs(i,1))<inputs(i,2);
        err(i) = LSep(w(1),w(2),w(3),inputs(i,1)) - inputs(i,2);
    end
    hold off
    
    correct = out == desOut;
    errp = err + 10000*correct;
    [c,index] = min(errp);
    if(c<0)
        dw = [-1 -inputs(index,1) -inputs(index,2)];
    else
        dw = [1 inputs(index,1) inputs(index,2)];
    end
    w = w - dw
    
end

