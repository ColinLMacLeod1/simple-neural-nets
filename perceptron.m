inputs = [0 0;0 1; 1 0; 1 1];
desOut = [0;0;0;1];
weights = [rand(1) rand(1) rand(1)];
LSep = @(w0,w1,w2,x1,x2) (-w1/w2)*x1 - w0/w1;
