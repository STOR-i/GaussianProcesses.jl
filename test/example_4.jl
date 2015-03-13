#Test out different covariance functions

using GaP

#Training data

x=[-4.0,-3.0,-1.0,0.0,2.0];

y = 2.0*x + 0.5*rand(5);

#Test data
xpred = [-5.0:0.1:5.0];

#Specify covariance function, not that default hyperparameters are l=1 and sigma²=1
mZero = mZERO()

lin = LIN(0.5)
#mPoly = mPoly(1,0.0,1.0)
se = SE()
gp = GP(x,y,mZero,lin)

plotGP(gp, xpred)


