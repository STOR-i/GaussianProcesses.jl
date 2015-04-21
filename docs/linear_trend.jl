#Test out different covariance functions

using Gadfly, GaP

#Training data

x=[-4.0,-3.0,-1.0,0.0,2.0];

y = 2.0*x + 0.5*rand(5);

#Test data
xpred = [-5.0:0.1:5.0];


mLin = MeanLin(0.5)
kern = SE(0.0,0.0)
gp = GP(x,y,mLin,kern)

plot(gp, xpred)


