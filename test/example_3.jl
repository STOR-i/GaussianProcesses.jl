# Optimize a gaussianprocess

using gaussianprocesses

#Training data

x=[-4.0,-3.0,-1.0,0.0,2.0];
y=[-2.0,0.0,1.0,2.0,-1.0];

#Specify covariance function, not that default hyperparameters are l=1 and sigma²=1

se = SE()
gp = GP(x,y,meanZero,se)
optimize!(gp, method=:bfgs, show_trace=true)

