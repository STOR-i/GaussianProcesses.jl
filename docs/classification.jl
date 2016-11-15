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


out = mcmc(gp)
