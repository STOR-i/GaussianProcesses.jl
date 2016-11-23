#This file gives a demo of how the GP package handles non-Gaussian data on a classification example

using PyPlot
using GaussianProcesses

X = rand(20)
X = sort(X)
y = sin(10*X)
y=convert(Vector{Bool}, y.>0)

#scatter(X,y)

#Select mean, kernel and likelihood function
mZero = MeanZero()                   #Zero mean function
kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)
lik = Bernoulli()

gp = GPMC{Bool}(X',vec(y),mZero,kern,lik)     

optimize!(gp)

xtest = collect(linspace(0.05,1,20))
fmean, fvar = predict(gp,xtest')

plot(xtest,fmean)
plot(xtest,sin(10*xtest))


conf = 1.96*sqrt(diag(fvar))
u = fmean + conf
l = fmean - conf

fig = PyPlot.figure("Gaussian Process Classification")
ax = fig[:add_subplot](1,1,1)
ax[:set_xlim](xmin=0, xmax=1)
ax[:set_ylim](ymin=minimum(l), ymax=maximum(u))

ax[:plot](xtest,fmean,label="Mean function")       #plot the mean function
ax[:scatter](gp.X,gp.y,marker="o",label="Observations",color="black")  #plot the observations
ax[:plot](xtest,l,label="Confidence Region",color="red")          
ax[:plot](xtest,u,color="red")
ax[:fill_between](xtest,u,l, facecolor="blue", alpha=0.25)


out = mcmc(gp,mcrange=Klara.BasicMCRange(nsteps=100000, thinning=20,burnin=10000))

fs = Vector(Float64,size(out,2))
for i in size(out,2)
    GaussianProcesses.set_params!(gp,out[:,i])
    fmean, fvar = predict(gp,xtest')
    fs[i] = rand(Distributions.MvNormal(fmean,fvar))
end    

# prev_ll=gp.ll
# prev_params=GaussianProcesses.get_params(gp)
# prev_dll = gp.dll

# dθ=[zeros(gp.nobsv);0.0;0.0;1.0]*1e-4
# GaussianProcesses.set_params!(gp, prev_params.+dθ)
# GaussianProcesses.ll!(gp)

# println("change in log likelihood: ", gp.ll-prev_ll)
# println("expected change in log likelihood: ", dot(prev_dll, dθ))
