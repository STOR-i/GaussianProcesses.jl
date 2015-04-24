import Gadfly

# For the 1D case plots the Gaussian process at the requested points WILL EXTEND TO 2D
function Gadfly.plot(gp::GP, x::Matrix{Float64}, CI::Float64=1.96, res::Int=50)
    d, n = size(x)
    if d==1
        mu, Sigma = predict(gp, x)
        conf = CI*sqrt(max(Sigma, 0.0))
        u = mu + conf
        l = mu - conf
        
        Gadfly.plot(Gadfly.layer(x=x, y=mu, ymin=l, ymax=u, Gadfly.Geom.line, Gadfly.Geom.ribbon),
                Gadfly.layer(x=gp.x,y=gp.y,Gadfly.Geom.point))
   elseif d==2
        xmin = minimum(x[1,:]); xmax = maximum(x[1,:])
        ymin = minimum(x[2,:]); ymax = maximum(x[2,:])
        
        sx = (xmax-xmin)/(res-1)
        sy = (ymax-ymin)/(res-1)
        A = Array(Float64,2,res^2)
        for (i,a) in enumerate(xmin:sx:xmax)
            for (j,b) in enumerate(ymin:sy:ymax)
                A[:,(i-1)*res+j] = [a,b]
            end
        end

        mu = predict(gp,A)[1]
        z= reshape(mu,res,res)
        Gadfly.plot(Gadfly.layer(z=z,x=[xmin:sx:xmax],y=[ymin:sy:ymax],Gadfly.Geom.contour),
                    Gadfly.layer(x=x[1,:],y=x[2,:],Gadfly.Geom.point))
   else error("Only 1D and 2D plots are permitted")
   end
end

#For the 1D case
Gadfly.plot(gp::GP,x::Vector{Float64}) = Gadfly.plot(gp,x')

