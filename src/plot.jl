# Plots a 1D or 2D GPE object.
#
# Keyword arguments:
#   - β: level of confidence interval for ribbon (1D only)
#   - obsv: plot observation points (1D only)
#   - var: plot predicted variance rather than mean (2D only)
@recipe function f(gp::GPE; β=0.95, obsv=true, var=false)
    @assert gp.dim ∈ (1,2)
    if gp.dim == 1
        xlims --> (minimum(gp.X), maximum(gp.X))
        xmin, xmax = d[:xlims]
        x = linspace(xmin, xmax, 100)
        μ, Σ = predict_f(gp, x)
        y = μ
        err = invΦ((1+β)/2)*sqrt.(Σ)
        
        @series begin
            seriestype := :path
            ribbon := err
            fillcolor --> :lightblue
            color --> :black
            x,y
        end
        if obsv
            @series begin
                seriestype := :scatter
                markershape := :circle
                markercolor := :black
                gp.X', gp.y
            end
        end
    else
        xlims --> (minimum(gp.X[1,:]), maximum(gp.X[1,:]))
        ylims --> (minimum(gp.X[2,:]), maximum(gp.X[2,:]))
        xmin, xmax = d[:xlims]
        ymin, ymax = d[:ylims]
        x = linspace(xmin, xmax, 50)
        y = linspace(ymin, ymax, 50)
        xgrid = repmat(x', 50, 1)
        ygrid = repmat(y, 1, 50)
        μ, Σ = predict_f(gp,[vec(xgrid)';vec(ygrid)'])
        if var
            zgrid  = reshape(Σ,50,50)
        else
            zgrid  = reshape(μ,50,50)
        end
        x, y, zgrid
    end
end
