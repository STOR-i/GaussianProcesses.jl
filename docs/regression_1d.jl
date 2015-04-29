# Plot basic Gaussian Process

using Winston, GaP

# Training data
x=[-4.0,-3.0,-1.0,0.0,2.0];
y=[-2.0,0.0,1.0,2.0,-1.0];


#Select mean and covariance function
mZero = MeanZero()                   #Zero mean function
kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)

gp = GP(x,y,mZero,kern)                      #Fit the GP
optimize!(gp,method=:bfgs,show_trace=true)   #Optimise the hyperparameters

# Predict the GP at test points
xpred = [-5.0:0.1:5.0];
mu, Sigma = predict(gp,xpred);

#Plot the data 
plot(gp)
