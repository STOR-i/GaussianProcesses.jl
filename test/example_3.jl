# Optimize a gaussianprocess

using gaussianprocesses
using Gadfly
import Gadfly.plot

# For the 1D case plots the Gaussian process at the requested points
function plot(gp::GP, x::Array{Float64})
    mu, Sigma = predict(gp, x)
    conf = 2*sqrt(max(diag(Sigma), 0.0))
    u = mu + conf
    l = mu - conf
   plot(layer(x=x, y=mu, ymin=l, ymax=u, Geom.line, Geom.ribbon),
        layer(x=gp.x,y=gp.y,Geom.point))
end

#Training data

x=[-4.0,-3.0,-1.0,0.0,2.0];
y=[-2.0,0.0,1.0,2.0,-1.0];

#Test data
xpred = [-5.0:0.1:5.0];

#Specify covariance function, not that default hyperparameters are l=1 and sigma²=1
se = SE()

gp = GP(x,y,meanZero,se)
plot(gp, xpred)
optimize!(gp)
plot(gp, xpred)
