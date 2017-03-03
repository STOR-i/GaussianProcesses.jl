#This file tests the GP model with a Student's t likelihood

using Gadfly
using GaussianProcesses

n = 20
X = linspace(-3,3,n)
sigma = 1.0
Y = X + sigma*rand(Distributions.TDist(3),n)

#plot the data
plot(x=X,y=Y,Geom.point)

#build the model
m = MeanZero()
k = Mat(3/2,0.0,0.0)
l = StuTLik(3,0.1)
gp = GPMC(X', vec(Y), m, k, l)

#set the priors (need a better interface)
GaussianProcesses.set_priors!(gp.lik,[Distributions.Normal(-2.0,4.0)])
GaussianProcesses.set_priors!(gp.k,[Distributions.Normal(-2.0,4.0),Distributions.Normal(-2.0,4.0)])

optimize!(gp)

#MCMC
samples = mcmc(gp;mcrange=Klara.BasicMCRange(nsteps=50000, thinning=10, burnin=10000))

plot(y=samples[end,:],Geom.line) #check MCMC mixing

########################################################
#Plot posterior samples
xtest = linspace(minimum(gp.X),maximum(gp.X),50);
ymean = [];
fsamples = [];
for i in 1:size(samples,2)
    GaussianProcesses.set_params!(gp,samples[:,i])
    GaussianProcesses.update_lpost!(gp)
    push!(ymean, predict(gp,xtest,obs=true)[1])
    push!(fsamples,rand(gp,xtest))
end

quant = Array(Float64,50,2);
for i in 1:50
    quant[i,:] = quantile(fsamples[i],[0.025,0.975])
end

plot(layer(x=xtest,y=mean(fsamples),ymin=quant[:,1],ymax=quant[:,2],Geom.line,Geom.ribbon),layer(x=vec(X),y=Y,Geom.point))

################################
#Predict
layers = []
for ym in ymean
    push!(layers, layer(x=xtest,y=ym,Geom.line))
end

plot(layers...,Guide.xlabel("X"),Guide.ylabel("y"))

sd = [std(ymean[i]) for i in 1:50]

plot(layer(x=xtest,y=mean(ymean),ymin=mean(ymean)-2*sd,ymax=mean(ymean)+2*sd,Geom.line,Geom.ribbon),layer(x=X,y=Y,Geom.point))

