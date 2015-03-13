#Playing around with the mean polynomial function

using GaP

#Training data

x=[-4.0,-3.0,-1.0,0.0,2.0];

#Create linear data
y = 0.5*x.^2 + 2.0*x + 0.5*rand(5)

#Test data
xpred = [-5.0:0.1:5.0];

#Specify mean and covariance function 
beta = [0.5,2.0]
mPoly = mPOLY(beta')
se = SE()
gp = GP(x,y,mPoly,se)
plotGP(gp, xpred)

