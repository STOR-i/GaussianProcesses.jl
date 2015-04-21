# Optimize a gaussianprocess

using GaP

# Training data

x=[-4.0,-3.0,-1.0,0.0,2.0];
y=[-2.0,0.0,1.0,2.0,-1.0];


# Specify mean and covariance function, not that default hyperparameters are l=1 and sigma²=1

mZero = MeanZero()
kern = SE(0.0,0.0)
gp = GP(x,y,mZero,kern)
optimize!(gp, method=:bfgs, show_trace=true)
