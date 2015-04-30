import Winston


function plot1D(gp::GP; clim::(Float64, Float64)=(minimum(gp.x), maximum(gp.x)), CI::Float64=1.96, res::Int=1000)

        sx = (clim[2]-clim[1])/(res-1)
        x=[clim[1]:sx:clim[2]]
        mu, Sigma = predict(gp, x)
        conf = CI*sqrt(Sigma)
        u = mu + conf
        l = mu - conf
        
        p=Winston.FramedPlot()
             Winston.add(p,Winston.Curve(x,l,color="red"))
             Winston.add(p,Winston.Curve(x,u,color="red"))
             Winston.add(p,Winston.FillBetween(x,l,x,u))
             Winston.add(p,Winston.Curve(x,mu,color="black"))
             Winston.add(p,Winston.Points(gp.x,gp.y,kind="filled circle"))
end


function plot2D(gp::GP; clim::(Float64, Float64, Float64, Float64) = (minimum(gp.x[1,:]), maximum(gp.x[1,:]),
                                                                   minimum(gp.x[2,:]), maximum(gp.x[2,:])),
                res::Int=50)
        sx = (clim[2]-clim[1])/(res-1)
        sy = (clim[4]-clim[3])/(res-1)
        A = Array(Float64,2,res^2)
        for (i,a) in enumerate(clim[1]:sx:clim[2])
            for (j,b) in enumerate(clim[3]:sy:clim[4])
                A[:,(i-1)*res+j] = [a,b]
            end
        end

        mu = predict(gp,A)[1]
        z= reshape(mu,res,res)
        Winston.imagesc(z, (minimum(z),0.6maximum(z)))
end

function Winston.plot(gp::GP; kwargs...)
    d, n = size(gp.x)
    if d>2
        error("Only 1D and 2D plots are permitted")
    elseif d==1
        plot1D(gp; kwargs...)
    elseif d==2
        plot2D(gp; kwargs...)
    end
end
