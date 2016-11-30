#This file gives a demo of how the GP package handles non-Gaussian data on a classification example

using PyPlot
using GaussianProcesses

X = rand(20)
X = sort(X)
y = sin(10*X)
y=convert(Vector{Bool}, y.>0)

scatter(X,y)

#Select mean, kernel and likelihood function
mZero = MeanZero()                   #Zero mean function
kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)
lik = Bernoulli()

gp = GPMC{Bool}(X',vec(y),mZero,kern,lik)     

optimize!(gp)


samples = mcmc(gp,mcrange=Klara.BasicMCRange(nsteps=100000, thinning=20,burnin=10000))

xtest = linspace(0,1,100);
fs = Vector(Float64,100,size(samples,2))
for i in size(samples,2)
    GaussianProcesses.set_params!(gp,samples[:,i])
    GaussianProcesses.update_ll!(gp)
    fmean, fvar = predict(gp,xtest)
    fs[:,i] = rand(Distributions.MvNormal(fmean,fvar))
end



