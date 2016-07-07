#Test out different covariance functions

using Gadfly, GaussianProcesses

#Training data

x=[-4.0,-3.0,-1.0,0.0,2.0];
y = 2.0*x + 0.5*rand(5);

#Test data
xpred = collect(-5.0:0.1:5.0);


mLin = MeanLin([0.5])  #Linear mean function
kern = SE(0.0,0.0)     #Squared exponential kernel function
gp = GP(x,y,mLin,kern) #Fit the GP

plot(gp)               #Plot the GP


