#This file gives a demo of how the GP package handles non-Gaussian data on a classification example

using Gadfly
using GaussianProcesses

srand(112233)
X = rand(20)
X = sort(X)
y = sin(10*X)
y=convert(Vector{Bool}, y.>0)

plot(x=X,y=y)

#Select mean, kernel and likelihood function
mZero = MeanZero()                   #Zero mean function
kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)
lik = BernLik()

gp = GPMC{Bool}(X',vec(y),mZero,kern,lik)     

optimize!(gp)
GaussianProcesses.set_priors!(gp.k,[Distributions.Normal(0.0,2.0),Distributions.Normal(0.0,2.0)])

#mcmc doesn't seem to mix well
samples = mcmc(gp;mcrange=Klara.BasicMCRange(nsteps=50000, thinning=10, burnin=10000))

plot(y=samples[end,:],Geom.line) #check MCMC mixing

xtest = linspace(minimum(gp.X),maximum(gp.X),50);
ymean = [];
fsamples = [];
for i in 1:size(samples,2)
    GaussianProcesses.set_params!(gp,samples[:,i])
    GaussianProcesses.update_ll!(gp)
    push!(ymean, predict(gp,xtest,obs=true)[1])
    push!(fsamples,rand(gp,xtest))
    #fsamples = hcat(fsamples,samp)
end

layers = []
for f in fsamples
    push!(layers, layer(x=xtest,y=f,Geom.line))
end

plot(layers...,Guide.xlabel("X"),Guide.ylabel("f"))


#######################
#Predict 

layers = []
for ym in ymean
    push!(layers, layer(x=xtest,y=ym,Geom.line))
end

plot(layers...,Guide.xlabel("X"),Guide.ylabel("y"))


plot(layer(x=xtest,y=mean(ymean),Geom.line),
     layer(x=X,y=y,Geom.point))

