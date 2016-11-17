#An example of non-Gaussian regression using exponential observations

using PyPlot
using GaussianProcesses

X = linspace(-3,3,20)
Y = [rand(Distributions.Exponential(sin(X[i]).^2)) for i in 1:20]

#build the model
k = Mat(3/2,1.0,1.0) 
l = Exponential()
gp = GPMC{Float64}(X', vec(Y), MeanZero(), k, l)

xtest = collect(linspace(-3.1,3.1,20))
fmean, fvar = predict(gp,xtest')

plot(Y)
plot(exp(fmean))

optimize!(gp)

samples = mcmc(gp;mcrange=Klara.BasicMCRange(nsteps=50000, burnin=10000))



plot(xtest,fmean)
plot(xtest,sin(10*xtest))
