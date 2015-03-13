#Plot basic Gaussian Process

using GaP

#Training data

x=[-4.0,-3.0,-1.0,0.0,2.0];
y=[-2.0,0.0,1.0,2.0,-1.0];

#Test data
xpred = [-5.0:0.1:5.0];

#Specify covariance function, not that default hyperparameters are l=1 and sigma²=1
mZero = mZERO()
se = SE()
gp = GP(x,y,mZero,se)
plotGP(gp, xpred)


