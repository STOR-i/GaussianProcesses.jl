#Plot basic Gaussian Process

using GaP
using Gadfly
import Gadfly.plot

# For the 1D case plots the Gaussian process at the requested points
function plotGP(gp::GP, x::Array{Float64})
    mu, Sigma = predict(gp, x)
    conf = 2*sqrt(max(diag(Sigma), 0.0))
    u = mu + conf
    l = mu - conf
   plot(layer(x=x, y=mu, ymin=l, ymax=u, Geom.line, Geom.ribbon),
        layer(x=gp.x,y=gp.y,Geom.point))
end

#Training data

x=[-4.0,-3.0,-1.0,0.0,2.0];

#Create linear data
y = 2.0*x + 0.5*rand(5)

#Test data
xpred = [-5.0:0.1:5.0];

#Specify mean and covariance function 
beta = [2.0]
mLin = mLIN(beta)

se = SE()
mZero = mZERO()

gp = GP(x,y,mLin,se)
predict(gp, xpred)
plotGP(gp, xpred)


gp = GP(x,y,mZero,se)
predict(gp, xpred)
plotGP(gp, xpred)

