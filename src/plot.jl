@userplot plot
@userplot contour

@recipe function f(gp::GPBase, β::Float64=0.95)
    xmin, xmax = minimum(gp.X), maximum(gp.X)
    sx = (xmax-xmin)/100
    x = xmin:sx:xmax
    
    linecolor --> :black
    legend := false
    # x := x
    xlim := (xmin,xmax)
    
    mu, sigma = predict_f(gp, x)
    y = mu
    err = invΦ((1+β)/2)*sigma
    
    @series begin
        seriestype := :path
        ribbon := err
        fillcolor := :lightblue
        x,y
    end

    @series begin
        seriestype := :path
        color --> :black
        x,y
    end

    @series begin
        seriestype := :scatter
        markershape := :circle
        markercolor := :black
        gp.X',gp.y
    end
end


@recipe function f(gp::GPBase)
    if size(gp.X)[1] !=2
        error("2d plot only")
    end
    
    xmin, xmax = (minimum(gp.X[1,:]), minimum(gp.X[2,:])), (minimum(gp.X[1,:]), maximum(gp.X[2,:]))
    x = linspace(xmin[1], xmax[1], 50)
    y = linspace(xmin[1], xmax[2], 50)
    xgrid = repmat(x', 50, 1 )
    ygrid = repmat(y, 1, 50 )
    
    mu = predict_f(gp,[vec(xgrid)';vec(ygrid)'])[1]
    zgrid  = reshape(mu,50,50)

    @series begin
        seriestype := :contour
        xgrid, ygrid, zgrid
    end

    @series begin
        seriestype := :scatter
        markershape := :circle
        markercolor := :black
        gp.X',gp.y
    end

end
