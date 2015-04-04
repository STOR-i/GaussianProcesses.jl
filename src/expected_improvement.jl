# Expected improvement function

function EI(gp::GP,x::Matrix{Float64})
    size(x,1) == gp.dim || throw(ArgumentError("Arguments have inconsistent dimensions"))
    #Fit the GP
    mu, Sigma = predict(gp, x)
    maxY = maximum(gp.y)
    
    #Calculate useful terms
    s = sqrt(max(diag(Sigma),0.0))
    y = mu - maxY
    ynorm = y./s 
    
    #Calculate the expected improvement
    ei = y .* Φ(ynorm) + s .* φ(ynorm)
    ei = max(0,ei)
    return(ei)    
end

# 1-D case
EI(gp::GP, x::Vector{Float64}) = EI(gp, x')



