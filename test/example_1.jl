using gaussianprocesses
using Gadfly
import Gadfly.plot

# For the 1D case plots the Gaussian process at the requested points
function plot(gp::GP, x::Array{Float64})
    exp, l, u = predict(gp, x)
    plot(x=x, y=exp, ymin=l, ymax=u, Geom.line, Geom.ribbon)
end


xpred = [-4.0,-3.0,-1.0, 0.0, 2.0]
y = [-2.0, 0.0, 1.0, 2.0, -1.0]

gp = GP(xpred,y,rbf)
predict(gp, xpred)
plot(gp, [-5:0.1:5])
