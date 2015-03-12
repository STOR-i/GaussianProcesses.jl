#Test expected improvement function

using GaP

#Training data
x = 2*π*rand(5);
y = sin(x) + 0.05*randn(5);

#Test data
xpred = [0.0:0.1:2*π];

#Specify covariance function, not that default hyperparameters are l=1 and sigma²=1
mZero = mZERO()
mat32 = MAT32()

gp = GP(x,y,mZero,mat32)
plotEI(gp, xpred)

