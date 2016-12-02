#An example of non-Gaussian regression using exponential observations

using Gadfly
using GaussianProcesses

n = 50
X = linspace(-3,3,n)
Y = [rand(Distributions.Exponential(sin(X[i]).^2)) for i in 1:n]

#build the model
k = Mat(3/2,0.0,0.0)
l = Exponential()
gp = GPMC{Float64}(X', vec(Y), MeanZero(), k, l)

optimize!(gp)
xtest = collect(linspace(-4.0,4.0,20));
fmean, fvar = predict(gp,xtest);


plot(layer(x=xtest,y=exp(fmean),ymin=exp(fmean-1.96sqrt(fvar)),ymax=exp(fmean+1.96sqrt(fvar)),Geom.line,Geom.ribbon),layer(x=X,y=Y,Geom.point))

#MCMC
samples = mcmc(gp;mcrange=Klara.BasicMCRange(nsteps=50000, burnin=10000))

plot(y=samples[end,:],Geom.line) #check MCMC mixing

#Plot posterior samples

xtest = linspace(-4,4,100);
fsamples = Array(Float64,100,size(samples,2));
for i in 1:size(samples,2)
    GaussianProcesses.set_params!(gp,samples[:,i])
    GaussianProcesses.update_ll!(gp)
    fmean, fvar = predict(gp,xtest)
    fsamples[:,i] = rand(Distributions.MvNormal(fmean,fvar))
end    

rateSamples = exp(fsamples);

fmean = mean(rateSamples,2); fvar = var(rateSamples,2);
plot(layer(x=xtest,y=exp(fmean),ymin=exp(fmean-1.96sqrt(fvar)),ymax=exp(fmean+1.96sqrt(fvar)),Geom.line,Geom.ribbon),layer(x=X,y=Y,Geom.point))

