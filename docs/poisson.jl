#An example of non-Gaussian regression using poisson observations

using Gadfly
using GaussianProcesses
srand(201216)


n = 20
X = linspace(-3,3,n)
f = 2*cos(2*X)
Y = [rand(Distributions.Poisson(exp(f[i]))) for i in 1:n]

#plot the data
#plot(x=X,y=Y,Geom.point)
#build the model
k = Mat(3/2,0.0,0.0)
l = PoisLik()
gp = GPMC(X', vec(Y), MeanZero(), k, l)
#set the priors (need a better interface)
GaussianProcesses.set_priors!(gp.k,[Distributions.Normal(-2.0,4.0),Distributions.Normal(-2.0,4.0)])

optimize!(gp)

#MCMC
samples = mcmc(gp;mcrange=Klara.BasicMCRange(nsteps=50000, thinning=10, burnin=10000))

plot(y=samples[end,:],Geom.line) #check MCMC mixing

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

plot(layer(x=xtest,y=mean(fsamples),Geom.line),
     layer(x=xtest,y=quant[:,1],Geom.line,Theme(default_color=colorant"red")),
     layer(x=xtest,y=quant[:,2],Geom.line,Theme(default_color=colorant"red")),
     layer(x=vec(X),y=f,Geom.point))

################################
#Predict

layers = []
for ym in ymean
    push!(layers, layer(x=xtest,y=ym,Geom.line))
end

plot(layers...,Guide.xlabel("X"),Guide.ylabel("y"))


plot(layer(x=xtest,y=mean(ymean),Geom.line),
     layer(x=X,y=Y,Geom.point))


