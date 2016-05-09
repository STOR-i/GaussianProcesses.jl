#Sample from the GP

using Gadfly, GaussianProcesses
import Gadfly.plot

srand(13579)
# Training data
n=10                 #number of training points
x = 2π * rand(n)              
y = cos(π*x) + 0.05*randn(n)

#Select mean and covariance function
mZero = MeanZero()                   #Zero mean function
kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)

gp = GP(x,y,mZero,kern,-2.0)      #Fit the GP, where -1.0 is the log Gaussian noise

path=sample(gp,10,collect(linspace(0,2*pi)))
plot(layer(x=collect(linspace(0,2*pi)),y=path[:,1],Geom.line),
     layer(x=collect(linspace(0,2*pi)),y=path[:,2],Geom.line,Theme(default_color=colorant"red")),
     layer(x=x,y=y,Geom.point))
