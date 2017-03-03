@recipe function f(gp::GP, β::Float64=0.95)
    xmin, xmax = minimum(gp.X), maximum(gp.X)
    sx = (xmax-xmin)/100
    x = xmin:sx:xmax
    
    linecolor --> :black
    legend := false
    # x := x
    xlim := (xmin,xmax)
    
    mu, sigma = predict(gp, x)
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
        markershape := :x
        markercolor := :black
        gp.X',gp.y
    end
end

