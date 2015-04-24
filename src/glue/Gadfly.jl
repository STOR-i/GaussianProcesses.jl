import Gadfly


function plot1D(gp::GP, c::(Float64, Float64), CI::Float64=1.96, res::Int=100)

        sx = (c[2]-c[1])/(res-1)
        x=[c[1]:sx:c[2]]
        mu, Sigma = predict(gp, x)
        conf = CI*sqrt(max(Sigma, 0.0))
        u = mu + conf
        l = mu - conf
        
        Gadfly.plot(Gadfly.layer(x=x, y=mu, ymin=l, ymax=u, Gadfly.Geom.line, Gadfly.Geom.ribbon),
                Gadfly.layer(x=gp.x,y=gp.y,Gadfly.Geom.point))
end


function plot2D(gp::GP, c::(Float64, Float64, Float64, Float64), res::Int=50)
        
        sx = (c[2]-c[1])/(res-1)
        sy = (c[4]-c[3])/(res-1)
        A = Array(Float64,2,res^2)
        for (i,a) in enumerate(c[1]:sx:c[2])
            for (j,b) in enumerate(c[3]:sy:c[4])
                A[:,(i-1)*res+j] = [a,b]
            end
        end

        mu = predict(gp,A)[1]
        z= reshape(mu,res,res)
        Gadfly.plot(z=z,x=[c[1]:sx:c[2]],y=[c[3]:sy:c[4]],Gadfly.Geom.contour)
end

function Gadfly.plot(gp::GP)
    d, n = size(gp.x)
    if d>2
        error("Only 1D and 2D plots are permitted")
    elseif d==1
        xmin = minimum(gp.x)
        xmax = maximum(gp.x)
        plot1D(gp, (xmin,xmax))
    elseif d==2
        xmin = minimum(gp.x[1,:]); xmax = maximum(gp.x[1,:])
        ymin = minimum(gp.x[2,:]); ymax = maximum(gp.x[2,:])
        plot2D(gp, (xmin,xmax,ymin,ymax))
    end
end



