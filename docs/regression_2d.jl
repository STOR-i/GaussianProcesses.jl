#Plot basic Gaussian Process

using Gadfly, GaussianProcesses

#Training data
d, n = 2, 50         #Dimension and number of observations
 
x = 2π * rand(d, n)                               
y = vec(sin(x[1,:]).*sin(x[2,:])) + 0.05*rand(n)
l = GaussLik(0.0)

mZero = MeanZero()                    #Zero mean function
kern = SEIso(0.0,0.0)        #Matern 5/2 ARD kernel with parameters log(l₁) = 0 log(l₂) = 0 and log(σ) = 0 and SE Iso kernel with parameters log(ℓ) = 0 and log(σ) = 0

gp = GPMC(x,y,mZero,kern,l)          # Fit the GP
optimize!(gp)                         # Optimize the hyperparameters

push!(gp, [[2.0; 3.0] [3.0, 4.0]], [2.0, 3.0])
push!(gp, [2.0; 3.0], 2.0)
plot(gp; clim=(-10.0, 10.0,-10.0,10.0)) #Plot the GP over range clim

