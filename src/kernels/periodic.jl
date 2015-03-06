# Periodic Function 

type PERIODIC <: Kernel
    ll::Float64      # Log of Length scale 
    lσ::Float64      # Log of signal std
    lp::Float64      # Log of period
    PERIODIC(l::Float64=1.0, σ::Float64=1.0, p::Float64=1.0) = new(log(l), log(σ), log(p))
end

function kern(periodic::PERIODIC, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(periodic.ll)
    sigma2 = exp(2*periodic.lσ)
    p      = exp(periodic.lp)

    K = sigma2*exp(-2/ell^2*sin(pi*norm(x-y)/p)^2)
    return K
end

params(periodic::PERIODIC) = exp(Float64[periodic.ll, periodic.lσ, periodic.lp])

num_params(periodic::PERIODIC) = 3

function set_params!(periodic::PERIODIC, hyp::Vector{Float64})
    length(hyp) == 3 || throw(ArgumentError("Periodic function has three parameters"))
    periodic.ll, periodic.lσ, periodic.lp = hyp
end

function grad_kern(periodic::PERIODIC, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(periodic.ll)
    sigma2 = exp(2*periodic.lσ)
    p      = exp(periodic.lp)
    
    dK_ell   = 4.0*sigma2*(sin(pi*norm(x-y)/p)/ell)^2*exp(-2/ell^2*sin(pi*norm(x-y)/p)^2)
    dK_sigma = 2.0*sigma2*exp(-2/ell^2*sin(pi*norm(x-y)/p)^2)
    dK_p     = 4.0/ell^2*sigma2*(pi*norm(x-y)/p)*sin(pi*norm(x-y)/p)*cos(pi*norm(x-y)/p)*exp(-2/ell^2*sin(pi*norm(x-y)/p)^2)
    dK_theta = [dK_ell,dK_sigma,dK_p]
    return dK_theta
end
