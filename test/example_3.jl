# Optimize a gaussianprocess

using gaussianprocesses

#Training data

x = 2*π*rand(5);
y = cos(x) + 0.5*randn(5);


#Specify covariance function, not that default hyperparameters are l=1 and sigma²=1
se = SE()

gp = GP(x,y,meanZero,se)
optimize!(gp)


