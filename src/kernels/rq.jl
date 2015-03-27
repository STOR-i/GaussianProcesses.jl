# Rational Quadratic Isotropic Covariance Function 

type RQ <: Kernel
    ll::Float64      # Log of length scale 
    lσ::Float64      # Log of signal std
    lα::Float64      # Log of shape parameter
    RQ(ll::Float64=0.0, lσ::Float64=0.0, lα::Float64=0.0) = new(ll, lσ, lα)
end

function kern(rq::RQ, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(rq.ll)
    sigma2 = exp(2*rq.lσ)
    alpha      = exp(rq.lα)

    K =  sigma2*(1+(norm(x-y)^2)/(2*alpha*ell^2)).^(-alpha)
    return K
end

params(rq::RQ) = Float64[rq.ll, rq.lσ, rq.lα]
num_params(rq::RQ) = 3

function set_params!(rq::RQ, hyp::Vector{Float64})
    length(hyp) == 3 || throw(ArgumentError("Rational Quadratic function has three parameters"))
    rq.ll, rq.lσ, rq.lα = hyp
end

function grad_kern(rq::RQ, x::Vector{Float64}, y::Vector{Float64})
    ell    = exp(rq.ll)
    sigma2 = exp(2*rq.lσ)
    alpha  = exp(rq.lα)
    
    dK_ell   = sigma2*((norm(x-y)^2)/ell^2)*(1+(norm(x-y)^2)/(2*alpha*ell^2))^(-alpha-1)
    dK_sigma = 2.0*sigma2*(1+(norm(x-y)^2)/(2*alpha*ell^2))^(-alpha)
    
    part     = (1+(norm(x-y)^2)/(2*alpha*ell^2))
    dK_alpha = sigma2*part^(-alpha)*((norm(x-y)^2)/(2*ell^2*part)-alpha*log(part))
    dK_theta = [dK_ell,dK_sigma,dK_alpha]
    return dK_theta
end
