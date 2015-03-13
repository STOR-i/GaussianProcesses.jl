#Polynomial covariance function (CURRENTLY BROKEN)
#See page 94 of Rasmussen and Williams

type POLY <: Kernel
    lc::Float64      # Log of constant
    lσ::Float64      # Log of signal std
    deg::Int64       # degree of polynomial
    POLY(c::Float64=0.0, σ::Float64=1.0, deg::Int64) = new(log(c), log(σ), deg)
end

function kern(poly::POLY, x::Vector{Float64}, y::Vector{Float64})
    c = exp(poly.lc)
    sigma2 = exp(2*poly.lσ)

    K = sigma2*(c+dot(x,y)).^poly.deg
    return K
end

params(poly::POLY) = exp(Float64[poly.lc, poly.lσ])

num_params(poly::POLY) = 2

function set_params!(poly::POLY, hyp::Vector{Float64})
    length(hyp) == 2 || throw(ArgumentError("Polynomial function has two parameters"))
    poly.lc, poly.lσ = hyp
end

function grad_kern(poly::POLY, x::Vector{Float64}, y::Vector{Float64})
    c = exp(poly.lc)
    sigma2 = exp(2*poly.lσ)
    
    dK_c   = c*poly.deg*sigma2*(c+dot(x,y)).^(poly.deg-1)
    dK_sigma = 2.0*sigma2*(c+dot(x,y)).^poly.deg
    dK_theta = [dK_c,dK_sigma]
    return dK_theta
end
