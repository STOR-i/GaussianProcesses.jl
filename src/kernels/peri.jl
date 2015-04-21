# Periodic Function 

@doc """
# Description
Constructor for the Periodic kernel (covariance)

k(x,x') = σ²exp(-2sin²(π|x-x'|/p)/ℓ²)
# Arguments:
* `ll::Vector{Float64}`: Log of length scale ℓ
* `lσ::Float64`        : Log of the signal standard deviation σ
* `lp::Float64`        : Log of the period
""" ->
type Peri <: Kernel
    ll::Float64      # Log of Length scale 
    lσ::Float64      # Log of signal std
    lp::Float64      # Log of period
    Peri(ll::Float64, lσ::Float64, lp::Float64) = new(ll, lσ, lp)
end

function kern(peri::Peri, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(peri.ll)
    sigma2 = exp(2*peri.lσ)
    p      = exp(peri.lp)

    K = sigma2*exp(-2/ell^2*sin(pi*norm(x-y)/p)^2)
    return K
end

get_params(peri::Peri) = Float64[peri.ll, peri.lσ, peri.lp]
num_params(peri::Peri) = 3

function set_params!(peri::Peri, hyp::Vector{Float64})
    length(hyp) == 3 || throw(ArgumentError("Periodic function has three parameters"))
    peri.ll, peri.lσ, peri.lp = hyp
end

function grad_kern(peri::Peri, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(peri.ll)
    sigma2 = exp(2*peri.lσ)
    p      = exp(peri.lp)
    
    dK_ell   = 4.0*sigma2*(sin(pi*norm(x-y)/p)/ell)^2*exp(-2/ell^2*sin(pi*norm(x-y)/p)^2)
    dK_sigma = 2.0*sigma2*exp(-2/ell^2*sin(pi*norm(x-y)/p)^2)
    dK_p     = 4.0/ell^2*sigma2*(pi*norm(x-y)/p)*sin(pi*norm(x-y)/p)*cos(pi*norm(x-y)/p)*exp(-2/ell^2*sin(pi*norm(x-y)/p)^2)
    dK_theta = [dK_ell,dK_sigma,dK_p]
    return dK_theta
end
