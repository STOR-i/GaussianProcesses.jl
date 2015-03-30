# Rational Quadratic ARD Covariance Function 

type RQard <: Kernel
    ll::Vector{Float64}      # Log of length scale 
    lσ::Float64              # Log of signal std
    lα::Float64              # Log of shape parameter
    dim::Int                 # Number of hyperparameters
    RQard(ll::Vector{Float64}, lσ::Float64=0.0, lα::Float64=0.0) = new(ll, lσ, lα, size(ll,1)+2)
end

function kern(rqArd::RQard, x::Vector{Float64}, y::Vector{Float64})
    ell    = exp(rqArd.ll)
    sigma2 = exp(2*rqArd.lσ)
    alpha  = exp(rqArd.lα)

    K =  sigma2*(1+0.5*(norm((x-y)./ell)^2)/alpha)^(-alpha)
    return K
end

params(rqArd::RQard) = [rqArd.ll, rqArd.lσ, rqArd.lα]
num_params(rqArd::RQard) = rqArd.dim

function set_params!(rqArd::RQard, hyp::Vector{Float64})
    length(hyp) == rqArd.dim || throw(ArgumentError("Rational Quadratic ARD function has $(rqArd.dim) parameters"))
    rqArd.ll = hyp[1:rqArd.dim-2]
    rqArd.lσ = hyp[rqArd.dim-1]
    rqArd.lα = hyp[rqArd.dim]
end

function grad_kern(rqArd::RQard, x::Vector{Float64}, y::Vector{Float64})
    ell    = exp(rqArd.ll)
    sigma2 = exp(2*rqArd.lσ)
    alpha  = exp(rqArd.lα)
    
    dK_ell   = sigma2*(((x-y)./ell).^2)*(1+0.5*(norm((x-y)./ell)^2)/alpha)^(-alpha-1)
    dK_sigma = 2.0*sigma2*(1+0.5*(norm((x-y)./ell)^2)/alpha)^(-alpha)
    
    part     = (1+0.5*(norm((x-y)./ell)^2)/alpha)
    dK_alpha = sigma2*part^(-alpha)*(0.5*(norm((x-y)./ell)^2)/part-alpha*log(part))
    dK_theta = [dK_ell,dK_sigma,dK_alpha]
    return dK_theta
end
