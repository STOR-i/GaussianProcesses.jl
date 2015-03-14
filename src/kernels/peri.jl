# Periodic Function 

type PERI <: Kernel
    ll::Float64      # Log of Length scale 
    lσ::Float64      # Log of signal std
    lp::Float64      # Log of period
    PERI(ll::Float64=0.0, lσ::Float64=0.0, lp::Float64=0.0) = new(ll, lσ, lp)
end

function kern(peri::PERI, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(peri.ll)
    sigma2 = exp(2*peri.lσ)
    p      = exp(peri.lp)

    K = sigma2*exp(-2/ell^2*sin(pi*norm(x-y)/p)^2)
    return K
end

params(peri::PERI) = Float64[peri.ll, peri.lσ, peri.lp]
num_params(peri::PERI) = 3

function set_params!(peri::PERI, hyp::Vector{Float64})
    length(hyp) == 3 || throw(ArgumentError("Periodic function has three parameters"))
    peri.ll, peri.lσ, peri.lp = hyp
end

function grad_kern(peri::PERI, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(peri.ll)
    sigma2 = exp(2*peri.lσ)
    p      = exp(peri.lp)
    
    dK_ell   = 4.0*sigma2*(sin(pi*norm(x-y)/p)/ell)^2*exp(-2/ell^2*sin(pi*norm(x-y)/p)^2)
    dK_sigma = 2.0*sigma2*exp(-2/ell^2*sin(pi*norm(x-y)/p)^2)
    dK_p     = 4.0/ell^2*sigma2*(pi*norm(x-y)/p)*sin(pi*norm(x-y)/p)*cos(pi*norm(x-y)/p)*exp(-2/ell^2*sin(pi*norm(x-y)/p)^2)
    dK_theta = [dK_ell,dK_sigma,dK_p]
    return dK_theta
end
