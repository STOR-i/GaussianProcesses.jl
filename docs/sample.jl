#Sample from the GP

using Gadfly, GaussianProcesses
import Gadfly.plot

srand(13579)
# Training data
n=10                 #number of training points
x = 2π * rand(n)              
y = cos(π*x) + 0.05*randn(n)

x=[-4.0,-3.0,-1.0,0.0,2.0];
y=[-2.0,0.0,1.0,2.0,-1.0];

#Select mean and covariance function
mZero = MeanZero()                   #Zero mean function
kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)

gp = GP(x,y,mZero,kern,-2.0)      #Fit the GP, where -1.0 is the log Gaussian noise
optimize!(gp)

range = collect(linspace(-5,5));
path=sample(gp,10,range)

plot(layer(x=range,y=path[:,1],Geom.line),
     layer(x=range,y=path[:,2],Geom.line,Theme(default_color=colorant"red")),
     layer(x=range,y=path[:,3],Geom.line,Theme(default_color=colorant"green")),
     layer(x=range,y=path[:,4],Geom.line,Theme(default_color=colorant"blue")),
     layer(x=range,y=path[:,5],Geom.line,Theme(default_color=colorant"yellow")),
     layer(x=x,y=y,Geom.point))

