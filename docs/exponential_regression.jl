#An example of non-Gaussian regression using exponential observations


using GaussianProcesses

X = linspace(-3,3,20)
Y = [rand(Distributions.Exponential(sin(X[i]).^2)) for i in 1:20]

#build the model
k = Mat(3/2,1.0,1.0) 
l = Exponential()
gp = GPMC{Real}(X', vec(Y), MeanZero(), k, l)


optimize!(gp)

samples = mcmc(gp)

