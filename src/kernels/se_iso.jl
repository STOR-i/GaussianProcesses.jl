# Squared Exponential Function with istropic distance

@doc """
# Description
Constructor for the isotropic Squared Exponential kernel (covariance)

k(x,x') = σ²exp(-(x-x')ᵀ(x-x')/2ℓ²)
# Arguments:
* `ll::Float64`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type SEIso <: Kernel
    ll::Float64      # Log of Length scale
    lσ::Float64      # Log of Signal std
    SEIso(ll::Float64, lσ::Float64) = new(ll,lσ)
end

function kern(se::SEIso, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(se.ll)
    sigma2 = exp(2*se.lσ)
    K = sigma2*exp(-0.5*sqeuclidean(x, y)/ell^2)
    #K = sigma2*exp(-0.5*norm(x-y)^2/ell^2)
    return K
end

get_params(se::SEIso) = Float64[se.ll, se.lσ]
num_params(se::SEIso) = 2

function set_params!(se::SEIso, hyp::Vector{Float64})
    length(hyp) == 2 || throw(ArgumentError("Squared exponential only has two parameters"))
    se.ll, se.lσ = hyp
end

function grad_kern(se::SEIso, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(se.ll)
    sigma2 = exp(2*se.lσ)
    
    dK_ell = sigma2*norm(x-y)^2/ell^2*exp(-0.5*norm(x-y)^2/ell^2)
    dK_sigma = 2.0*sigma2*exp(-0.5*norm(x-y)^2/ell^2)
    
    dK_theta = [dK_ell,dK_sigma]
    return dK_theta
end

function crossKern(X::Matrix{Float64}, se::SEIso)
    ell = exp(se.ll)
    sigma2 = exp(2*se.lσ)
    R = pairwise(SqEuclidean(), X)
    broadcast!(/, R, R, -2*ell^2)
    map!(exp, R, R)
    broadcast!(*, R, R, 2.0*sigma2)
    return R
end
