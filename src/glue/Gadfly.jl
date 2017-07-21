import Gadfly

function plot1D(gp::GPBase; clim::Tuple{Float64, Float64}=(minimum(gp.X), maximum(gp.X)), CI::Float64=1.96, res::Int=1000)

        sx = (clim[2]-clim[1])/(res-1)
        x=collect(clim[1]:sx:clim[2])
        mu, Sigma = predict_f(gp, x)
        conf = CI*sqrt(Sigma)
        u = mu + conf
        l = mu - conf
        
        Gadfly.plot(Gadfly.layer(x=x, y=mu, ymin=l, ymax=u, Gadfly.Geom.line, Gadfly.Geom.ribbon),
                Gadfly.layer(x=gp.X,y=gp.y,Gadfly.Geom.point))
end


function plot2D(gp::GPBase; clim::Tuple{Float64, Float64, Float64, Float64} = (minimum(gp.X[1,:]), maximum(gp.X[1,:]),
                                                                   minimum(gp.X[2,:]), maximum(gp.X[2,:])),
                res::Int=50)
        sx = (clim[2]-clim[1])/(res-1)
        sy = (clim[4]-clim[3])/(res-1)
        A = Array(Float64,2,res^2)
        for (i,a) in enumerate(clim[1]:sx:clim[2])
            for (j,b) in enumerate(clim[3]:sy:clim[4])
                A[:,(i-1)*res+j] = [a,b]
            end
        end

        mu = predict_f(gp,A)[1]
        z= reshape(mu,res,res)
        Gadfly.plot(z=z,x=collect(clim[1]:sx:clim[2]),y=collect(clim[3]:sy:clim[4]),Gadfly.Geom.contour)
end

function Gadfly.plot(gp::GPBase; kwargs...)
    d, n = size(gp.X)
    if d>2
        error("Can only plot one or two dimensional Gaussian processes")
    elseif d==1
        plot1D(gp; kwargs...)
    elseif d==2
        plot2D(gp; kwargs...)
    end
end
