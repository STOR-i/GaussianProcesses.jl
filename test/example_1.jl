#Plot basic Gaussian Process

using gaussianprocesses
using Distributions
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
x = 2*π*rand(5);
y = cos(x) + rand(Normal(0,0.5),5);

#Test data
xpred = [-2*π:0.1:2*π];

#Specify covariance function, not that default hyperparameters are l=1 and sigma²=1
mat32 = MAT32()

gp = GP(x,y,meanZero,mat32)
predict(gp, xpred)
plot(gp, xpred)
