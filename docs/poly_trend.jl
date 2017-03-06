#Playing around with the mean polynomial function

using Gadfly, GaussianProcesses

#Training data
x=[-4.0,-3.0,-1.0,0.0,2.0];

# Create quadratic function
y = 0.5*x.^2 + 2.0*x + 0.5*rand(5)

#Test data
xpred = collect(-5.0:0.1:5.0);

#Specify mean and covariance function 
beta = [0.5 2.0]
mPoly = MeanPoly(beta)
kern = SE(0.0,0.0)
gp = GPE(x,y,mPoly,kern)  #Fit the GP

plot(gp)   #Plot the GP
