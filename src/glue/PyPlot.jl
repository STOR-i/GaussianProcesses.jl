import PyPlot


function plot1D(gp::GP; clim::Tuple{Float64, Float64}=(minimum(gp.x), maximum(gp.x)), CI::Float64=1.96, res::Int=1000)

    sx = (clim[2]-clim[1])/(res-1)
    x=collect(clim[1]:sx:clim[2])
    mu, Sigma = predict(gp, x)
    conf = CI*sqrt(Sigma)
    u = mu + conf
    l = mu - conf
    
    fig = PyPlot.figure("Gaussian Process Regression")
    ax = fig[:add_subplot](1,1,1)
    ax[:set_xlim](xmin=clim[1], xmax=clim[2])
    ax[:set_ylim](ymin=minimum(l), ymax=maximum(u))

    ax[:plot](x,mu,label="Mean function")       #plot the mean function
    ax[:scatter](gp.x,gp.y,marker="o",label="Observations",color="black")  #plot the observations
    ax[:plot](x,l,label="Confidence Region",color="red")            # plot the confidence region
    ax[:plot](x,u,color="red")
    PyPlot.legend(loc = "best", fontsize=15);
end


# function plot2D(gp::GP; clim::Tuple{Float64, Float64, Float64, Float64} = (minimum(gp.x[1,:]), maximum(gp.x[1,:]),
#                                                                    minimum(gp.x[2,:]), maximum(gp.x[2,:])),
#                 res::Int=50)
#         sx = (clim[2]-clim[1])/(res-1)
#         sy = (clim[4]-clim[3])/(res-1)
#         A = Array(Float64,2,res^2)
#         for (i,a) in enumerate(clim[1]:sx:clim[2])
#             for (j,b) in enumerate(clim[3]:sy:clim[4])
#                 A[:,(i-1)*res+j] = [a,b]
#             end
#         end

#         mu = predict(gp,A)[1]
#         z= reshape(mu,res,res)
#         Winston.imagesc(z, (minimum(z),0.6maximum(z)))
# end

function PyPlot.plot(gp::GP; kwargs...)
    d, n = size(gp.x)
    if d>2
        error("Only 1D and 2D plots are permitted")
    elseif d==1
        plot1D(gp; kwargs...)
    elseif d==2
        plot2D(gp; kwargs...)
    end
end
