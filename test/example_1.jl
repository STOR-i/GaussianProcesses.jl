#Plot basic Gaussian Process

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


x = [-4.0,-3.0,-1.0, 0.0, 2.0]
y = [-2.0, 0.0, 1.0, 2.0, -1.0]
xpred = [-5:0.1:5]

gp = GP(x,y,meanZero,mat32)
predict(gp, xpred)
plot(gp, xpred)
