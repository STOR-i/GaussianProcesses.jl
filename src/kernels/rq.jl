# Rational Quadratic Covariance Function 

type RQ <: Kernel
    ll::Float64      # Log of length scale 
    lσ::Float64      # Log of signal std
    lα::Float64      # Log of shape parameter
    RQ(l::Float64=1.0, σ::Float64=1.0, α::Float64=1.0) = new(log(l), log(σ), log(α))
end

function kern(rq::RQ, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(rq.ll)
    sigma2 = exp(2*rq.lσ)
    p      = exp(rq.lα)

    K = 
    return K
end

params(rq::RQ) = exp(Float64[rq.ll, rq.lσ, rq.lα])

num_params(rq::RQ) = 3

function set_params!(rq::RQ, hyp::Vector{Float64})
    length(hyp) == 3 || throw(ArgumentError("Rational Quadratic function has three parameters"))
    rq.ll, rq.lσ, rq.lα = hyp
end

function grad_kern(rq::RQ, x::Vector{Float64}, y::Vector{Float64})
    ell    = exp(rq.ll)
    sigma2 = exp(2*rq.lσ)
    alpha  = exp(rq.lα)
    
    dK_ell   = 
    dK_sigma = 
    dK_alpha = 
    dK_theta = [dK_ell,dK_sigma,dK_alpha]
    return dK_theta
end
