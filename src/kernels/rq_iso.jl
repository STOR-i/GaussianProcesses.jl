# Rational Quadratic Isotropic Covariance Function 

@doc """
# Description
Constructor for the isotropic Rational Quadratic kernel (covariance)

k(x,x') = σ²(1+(x-x')ᵀ(x-x')/2αℓ²)^{-α}
# Arguments:
* `ll::Float64`: Log of length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
* `lα::Float64`: Log of shape parameter α
""" ->
type RQIso <: Kernel
    ll::Float64      # Log of length scale 
    lσ::Float64      # Log of signal std
    lα::Float64      # Log of shape parameter
    RQIso(ll::Float64, lσ::Float64, lα::Float64) = new(ll, lσ, lα)
end

function kern(rq::RQIso, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(rq.ll)
    sigma2 = exp(2*rq.lσ)
    alpha      = exp(rq.lα)
    K =  sigma2*(1+sqeuclidean(x, y)/(2*alpha*ell^2)).^(-alpha)
    return K
end

get_params(rq::RQIso) = Float64[rq.ll, rq.lσ, rq.lα]
num_params(rq::RQIso) = 3

function set_params!(rq::RQIso, hyp::Vector{Float64})
    length(hyp) == 3 || throw(ArgumentError("Rational Quadratic function has three parameters"))
    rq.ll, rq.lσ, rq.lα = hyp
end

function grad_kern(rq::RQIso, x::Vector{Float64}, y::Vector{Float64})
    ell    = exp(rq.ll)
    sigma2 = exp(2*rq.lσ)
    alpha  = exp(rq.lα)
    dxy2 = sqeuclidean(x,y)
    
    dK_ell   = sigma2*((dxy2)/ell^2)*(1+(dxy2)/(2*alpha*ell^2))^(-alpha-1)
    dK_sigma = 2.0*sigma2*(1+(dxy2)/(2*alpha*ell^2))^(-alpha)
    
    part     = (1+(dxy2)/(2*alpha*ell^2))
    dK_alpha = sigma2*part^(-alpha)*((dxy2)/(2*ell^2*part)-alpha*log(part))
    dK_theta = [dK_ell,dK_sigma,dK_alpha]
    return dK_theta
end
