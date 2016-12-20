#This file gives a demo of how the GP package handles non-Gaussian data on a classification example

using Gadfly
using GaussianProcesses

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
GaussianProcesses.set_priors!(gp.k,[Distributions.Normal(-2.0,4.0),Distributions.Normal(-2.0,4.0)])

#mcmc doesn't seem to mix well
samples = mcmc(gp;mcrange=Klara.BasicMCRange(nsteps=50000, thinning=10, burnin=10000))

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
#######################
#Predict 

xtest = linspace(0,1,10);
ymean, yvar = predict(gp,xtest;obs=true)
plot(layer(x=xtest,y=ymean,Geom.point,Theme(default_color=color("red"))),
     layer(x=vec(X),y=y,Geom.point,Theme(default_color=color("blue"))))
