#Plot basic Gaussian Process

using Gadfly, GaussianProcesses

#Training data
d, n = 2, 50         #Dimension and number of observations
 
x = 2π * rand(d, n)                               
y = vec(sin(x[1,:]).*sin(x[2,:])) + 0.05*rand(n) 

mZero = MeanZero()                    #Zero mean function
#kern = Mat(5/2,[0.0,0.0],0.0) + SE(0.0,0.0)        #Matern 5/2 ARD kernel with parameters log(l₁) = 0 log(l₂) = 0 and log(σ) = 0 and SE Iso kernel with parameters log(ℓ) = 0 and log(σ) = 0

kern = SE(0.0, 0.0)

gp = GP(x,y,mZero,kern,-2.0)          # Fit the GP
optimize!(gp)                         # Optimize the hyperparameters

push!(gp, [[2.0; 3.0] [3.0, 4.0]], [2.0, 3.0])
push!(gp, [2.0; 3.0], 2.0)
plot(gp; clim=(-10.0, 10.0,-10.0,10.0)) #Plot the GP over range clim

