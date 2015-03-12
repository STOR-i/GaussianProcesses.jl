using Gadfly
import Gadfly.plot

# For the 1D case plots the Gaussian process at the requested points WILL EXTEND TO 2D
function plotGP(gp::GP, x::Array{Float64}, CI::Float64=1.96)
    mu, Sigma = predict(gp, x)
    conf = CI*sqrt(max(diag(Sigma), 0.0))
    u = mu + conf
    l = mu - conf
   plot(layer(x=x, y=mu, ymin=l, ymax=u, Geom.line, Geom.ribbon),
        layer(x=gp.x,y=gp.y,Geom.point))
end


# For the 1D Gaussian process at the requested points with the expected improvement
function plotEI(gp::GP, x::Array{Float64},CI::Float64=1.96)
    mu, Sigma = predict(gp, x)
    conf = CI*sqrt(max(diag(Sigma), 0.0))
    u = mu + conf
    l = mu - conf
    ei = EI(gp,x)   #Calculate the expected improvement
    p1 = plot(layer(x=x, y=mu, ymin=l, ymax=u, Geom.line, Geom.ribbon),layer(x=gp.x,y=gp.y,Geom.point))
    p2 = plot(x=x, y=ei,Geom.line)
    p1
    draw(PDF("p1and2.pdf", 6inch, 6inch), vstack(p1,p2))
end
