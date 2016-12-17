#An example of non-Gaussian regression using exponential observations

using Gadfly
using GaussianProcesses

n = 20
X = linspace(-3,3,n)
Y = [rand(Distributions.Poisson(sin(X[i]).^2)) for i in 1:n]

#plot the data
plot(x=X,y=Y,Geom.point)

#build the model
k = Mat(3/2,0.0,0.0)
l = PoisLik()
gp = GPMC{Int64}(X', vec(Y), MeanZero(), k, l)

#set the priors (need a better interface)
GaussianProcesses.set_priors!(gp.k,[Distributions.Normal(-2.0,4.0),Distributions.Normal(-2.0,4.0)])


optimize!(gp)

#MCMC
samples = mcmc(gp;mcrange=Klara.BasicMCRange(nsteps=50000, thinning=10, burnin=10000))

plot(y=samples[end,:],Geom.line) #check MCMC mixing

#Plot posterior samples

xtest = linspace(-4,4,100);
fsamples = Array(Float64,100);
for i in 1:size(samples,2)
    GaussianProcesses.set_params!(gp,samples[:,i])
    GaussianProcesses.update_ll!(gp)
    samp = rand(gp,xtest,5) 
    fsamples = hcat(fsamples,samp)
end    


rateSamples = exp(fsamples)
fmean = mean(rateSamples,2); 

quant = Array(Float64,100,2);
for i in 1:100
    quant[i,:] = quantile(rateSamples[i,:],[0.05,0.95])
end

plot(layer(x=xtest,y=fmean,ymin=quant[:,1],ymax=quant[:,2],Geom.line,Geom.ribbon),layer(x=vec(X),y=Y,Geom.point))

##############################
#need to check the posterior as well
function test(hyp::Vector{Float64})
    GaussianProcesses.set_params!(gp, hyp)
    GaussianProcesses.update_ll!(gp)
    return gp.ll
end

v = GaussianProcesses.get_params(gp);
GaussianProcesses.update_ll_and_dll!(gp,Array(Float64,gp.nobsv,gp.nobsv))
Calculus.gradient(test,v)
gp.dll
