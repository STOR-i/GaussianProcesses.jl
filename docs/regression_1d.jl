# Plot basic Gaussian Process

using PyPlot, GaussianProcesses

srand(13579)
# Training data
n=10                 #number of training points
x = 2π * rand(n)              
y = sin(x) + 0.05*randn(n)

#Select mean and covariance function
mZero = MeanZero()                   #Zero mean function
kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)

gp = GP(x,y,mZero,kern,-1.0)      #Fit the GP, where -1.0 is the log Gaussian noise
plot(gp)                          #Plot the GP

optimize!(gp; method=Optim.BFGS())   #Optimise the hyperparameters

plot(gp)   #Plot the GP after the hyperparameters have been optimised 

# Add observation
push!(gp, [1.0], [2.0])






