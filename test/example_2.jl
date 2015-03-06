#Test expected improvement function

using GaP
using Gadfly
import Gadfly.plot

# For the 1D case plots the Gaussian process at the requested points with the expected improvement
function plot(gp::GP, x::Array{Float64})
    mu, Sigma = predict(gp, x)
    conf = 2*sqrt(max(diag(Sigma), 0.0))
    u = mu + conf
    l = mu - conf
    ei = EI(gp,x)   #Calculate the expected improvement
    p1 = plot(layer(x=x, y=mu, ymin=l, ymax=u, Geom.line, Geom.ribbon),layer(x=gp.x,y=gp.y,Geom.point))
    p2 = plot(x=x, y=ei,Geom.line)
    p1
    draw(PDF("p1and2.pdf", 6inch, 6inch), vstack(p1,p2))
end


#Training data
x = 2*π*rand(5);
y = cos(x) + randn(5);

#Test data
xpred = [-2*π:0.1:2*π];

#Specify covariance function, not that default hyperparameters are l=1 and sigma²=1
mat32 = MAT32()

gp = GP(x,y,meanZero,mat32)
plot(gp, xpred)

