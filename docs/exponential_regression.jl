#An example of non-Gaussian regression using exponential observations

using Gadfly
using GaussianProcesses

n = 20
X = linspace(-3,3,n)
Y = [rand(Distributions.Exponential(sin(X[i]).^2)) for i in 1:n]

#build the model
k = Mat(3/2,0.0,0.0) #+ Const(1.0)
l = ExpLik()
gp = GPMC(X', vec(Y), MeanZero(), k, l)

#set the priors (need a better interface)
GaussianProcesses.set_priors!(gp.k,[Distributions.Normal(-2.0,4.0),Distributions.Normal(-2.0,4.0)])

optimize!(gp)
xtest = collect(linspace(-4.0,4.0,20));
fmean, fvar = predict(gp,xtest);

plot(layer(x=xtest,y=exp(fmean),ymin=exp(fmean-1.96sqrt(fvar)),ymax=exp(fmean+1.96sqrt(fvar)),Geom.line,Geom.ribbon),layer(x=X,y=Y,Geom.point))

#MCMC
samples = mcmc(gp;mcrange=Klara.BasicMCRange(nsteps=50000, thinning=10, burnin=10000))

plot(y=samples[:,end],Geom.line) #check MCMC mixing

#Plot posterior samples

xtest = linspace(-4,4,100);
fsamples = Array(Float64,100);
for i in 1:size(samples,2)
    GaussianProcesses.set_params!(gp,samples[:,i])
    GaussianProcesses.update_ll!(gp)
    samp = rand(gp,xtest,50) 
    fsamples = hcat(fsamples,samp)
end    
fsamples = fsamples[:,2:end]

rateSamples = exp(fsamples)
fmean = mean(rateSamples,2); 

quant = Array(Float64,100,2);
for i in 1:100
    quant[i,:] = quantile(rateSamples[i,:],[0.05,0.95])
end

plot(layer(x=xtest,y=fmean,ymin=quant[:,1],ymax=quant[:,2],Geom.line,Geom.ribbon),layer(x=vec(X),y=Y,Geom.point))
