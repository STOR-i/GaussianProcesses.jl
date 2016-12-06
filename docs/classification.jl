#This file gives a demo of how the GP package handles non-Gaussian data on a classification example

using Gadfly
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


samples = mcmc(gp,mcrange=Klara.BasicMCRange(nsteps=100000, thinning=50,burnin=10000))

plot(y=samples[end,:],Geom.line) #check MCMC mixing

xtest = linspace(0,1,50);
fsamples = Array(Float64,50);
for i in 1:size(samples,2)
    GaussianProcesses.set_params!(gp,samples[:,i])
    GaussianProcesses.update_ll!(gp)
    samp = rand(gp,xtest,5) 
    fsamples = hcat(fsamples,samp)
end

fsamples = fsamples[:,2:end]
fmean = mean(fsamples,2); 
plot(layer(x=xtest,y=fmean,Geom.line),layer(x=vec(X),y=y,Geom.point))
#plot(layer(x=xtest,y=convert(Vector{Bool},fmean[:].>0),Geom.line),layer(x=vec(X),y=y,Geom.point))



