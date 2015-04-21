import Gadfly

# For the 1D case plots the Gaussian process at the requested points WILL EXTEND TO 2D
function Gadfly.plot(gp::GP, x::Matrix{Float64}, CI::Float64=1.96)
    d, n = size(x)
    if d==1
    mu, Sigma = predict(gp, x)
    conf = CI*sqrt(max(Sigma, 0.0))
    u = mu + conf
    l = mu - conf
    Gadfly.plot(Gadfly.layer(x=x, y=mu, ymin=l, ymax=u, Gadfly.Geom.line, Gadfly.Geom.ribbon),
                Gadfly.layer(x=gp.x,y=gp.y,Gadfly.Geom.point))
   elseif d==2
        println("2D not yet implemented")
   else error("Only 1D and 2D plots are permitted")
   end
end

#For the 1D case
Gadfly.plot(gp::GP,x::Vector{Float64}) = Gadfly.plot(gp,x')

