import Gadfly

# For the 1D case plots the Gaussian process at the requested points WILL EXTEND TO 2D
function Gadfly.plot(gp::GP, x::Array{Float64}, CI::Float64=1.96)
    mu, Sigma = predict(gp, x)
    conf = CI*sqrt(max(diag(Sigma), 0.0))
    u = mu + conf
    l = mu - conf
    Gadfly.plot(Gadfly.layer(x=x, y=mu, ymin=l, ymax=u, Gadfly.Geom.line, Gadfly.Geom.ribbon),
                Gadfly.layer(x=gp.x,y=gp.y,Gadfly.Geom.point))
end


# For the 1D Gaussian process at the requested points with the expected improvement
function plotEI(gp::GP, x::Array{Float64},CI::Float64=1.96)
    mu, Sigma = predict(gp, x)
    conf = CI*sqrt(max(diag(Sigma), 0.0))
    u = mu + conf
    l = mu - conf
    ei = EI(gp,x)   #Calculate the expected improvement
    p1 = Gadfly.plot(x=x, y=ei,Gadfly.Geom.line)
    p1
end
