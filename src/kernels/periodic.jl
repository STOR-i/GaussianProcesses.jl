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
type Periodic <: Kernel
    ll::Float64      # Log of Length scale 
    lσ::Float64      # Log of signal std
    lp::Float64      # Log of period
    Periodic(ll::Float64, lσ::Float64, lp::Float64) = new(ll, lσ, lp)
end

function kern(peri::Periodic, x::Vector{Float64}, y::Vector{Float64})
    ℓ2 = exp(2.0*peri.ll)
    σ2 = exp(2.0*peri.lσ)
    p = exp(peri.lp)
    σ2*exp(-2.0/ℓ2*sin(π*euclidean(x,y)/p)^2)
end

get_params(peri::Periodic) = Float64[peri.ll, peri.lσ, peri.lp]
num_params(peri::Periodic) = 3

function set_params!(peri::Periodic, hyp::Vector{Float64})
    length(hyp) == 3 || throw(ArgumentError("Periodic function has only three parameters"))
    peri.ll, peri.lσ, peri.lp = hyp
end

function grad_kern(peri::Periodic, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(peri.ll)
    sigma2 = exp(2*peri.lσ)
    p      = exp(peri.lp)
    dxy = euclidean(x,y)
    
    dK_ell   = 4.0*sigma2*(sin(pi*dxy/p)/ell)^2*exp(-2/ell^2*sin(pi*dxy/p)^2)
    dK_sigma = 2.0*sigma2*exp(-2/ell^2*sin(pi*dxy/p)^2)
    dK_p     = 4.0/ell^2*sigma2*(pi*dxy/p)*sin(pi*dxy/p)*cos(pi*dxy/p)*exp(-2/ell^2*sin(pi*dxy/p)^2)
    dK_theta = [dK_ell,dK_sigma,dK_p]
    return dK_theta
end

# This makes crossKern slower for some reason...

## function crossKern(X::Matrix{Float64}, peri::Periodic)
##     ℓ2 = exp(2.0*peri.ll)
##     σ2 = exp(2.0*peri.lσ)
##     p = exp(peri.lp)

##     R = pairwise(Euclidean(), X)
##     broadcast!(*, R, R, π/p)
##     map!(sin, R, R)
##     broadcast!(^, R, R, 2)
##     broadcast!(*, R, R, -2.0/ℓ2)
##     ## R^=2
##     ## R *= (-2.0/ℓ2)
##     map!(exp, R, R)
##     R *= σ2
## end
