# Plot basic Gaussian Process

using Gadfly, GaP

# Training data
x=[-4.0,-3.0,-1.0,0.0,2.0];
y=[-2.0,0.0,1.0,2.0,-1.0];


#Select mean and covariance function
mZero = MeanZero()                   #Zero mean function
kern = SE(5,0.0,0.0)                #Matern 5/2

gp = GP(x,y,mZero,kern)
optimize!(gp,method=:bfgs,show_trace=true)

# Predic the GP at test points
xpred = [-5.0:0.1:5.0];
mu, Sigma = predict(gp,xpred);

#Plot the data 
plot(gp)
