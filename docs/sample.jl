#Sample from the GP

using Gadfly, GaussianProcesses
import Gadfly.plot

srand(13579)

#Select mean and covariance function
mZero = MeanZero()                   # Zero mean function
kern = SE(0.0,0.0)                   # Sqaured exponential kernel (note that hyperparameters are on the log scale)

#First sample from GP prior
gp = GP(m=mZero,k=kern)     

range = collect(linspace(-5,5));
prior=rand(gp,range, 10)

plot(layer(x=range,y=prior[:,1],Geom.line), #plot sample paths of the prior
     layer(x=range,y=prior[:,2],Geom.line,Theme(default_color=colorant"red")),
     layer(x=range,y=prior[:,3],Geom.line,Theme(default_color=colorant"green")),
     layer(x=range,y=prior[:,4],Geom.line,Theme(default_color=colorant"blue")),
     layer(x=range,y=prior[:,5],Geom.line,Theme(default_color=colorant"yellow")))


# Training data
x=[-4.0,-3.0,-1.0,0.0,2.0];
y=[-2.0,0.0,1.0,2.0,-1.0];

gp = GP(x,y,mZero,kern)   

range = collect(linspace(-5,5));
post=rand(gp,range, 10)

plot(layer(x=range,y=post[:,1],Geom.line), #plot sample paths of the posterior
     layer(x=range,y=post[:,2],Geom.line,Theme(default_color=colorant"red")),
     layer(x=range,y=post[:,3],Geom.line,Theme(default_color=colorant"green")),
     layer(x=range,y=post[:,4],Geom.line,Theme(default_color=colorant"blue")),
     layer(x=range,y=post[:,5],Geom.line,Theme(default_color=colorant"yellow")),
     layer(x=x,y=y,Geom.point, Theme(default_color=colorant"black",default_point_size=4pt)))

