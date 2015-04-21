# Plot basic Gaussian Process

using Gadfly, GaP

# Training data

x=[-4.0,-3.0,-1.0,0.0,2.0];
y=[-2.0,0.0,1.0,2.0,-1.0];

# Test data
xpred = [-5.0:0.1:5.0];


mZero = MeanZero()
kern = SE(0.0,0.0)
gp = GP(x,y,mZero,kern)

# For plotting must have loading Gadfly before GaP or use initialisation function
plot(gp, xpred)
