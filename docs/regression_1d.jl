# Plot basic Gaussian Process

using Gadfly, GaP

# Training data

x=[-4.0,-3.0,-1.0,0.0,2.0];
y=[-2.0,0.0,1.0,2.0,-1.0];

# Test data
xpred = [-5.0:0.1:5.0];


mZero = MeanZero()
kern = Mat(5,0.0,0.0)*Mat(3,0.0,0.0)*SE(0.0,0.0)*RQ(0.0,0.0,0.0) + SE(0.0,0.0)

gp = GP(x,y,mZero,kern)
optimize!(gp,method=:bfgs,show_trace=true)

plot(gp, xpred)
