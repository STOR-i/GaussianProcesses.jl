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
    ℓ2 = exp(2.0*se.ll)
    σ2 = exp(2*se.lσ)

    dxy2 = sqeuclidean(x,y)
    exp_dist = exp(-0.5*dxy2/ℓ2)
    
    dK_ell = σ2*dxy2/ℓ2*exp_dist
    dK_sigma = 2.0*σ2*exp_dist
    
    dK_theta = [dK_ell,dK_sigma]
    return dK_theta
end

function crossKern(X::Matrix{Float64}, se::SEIso)
    d, nobsv = size(X)
    ℓ2 = exp(se.ll)
    σ2 = exp(2*se.lσ)
    R = pairwise(SqEuclidean(), X)
    for i in 1:nobsv, j in 1:i
        @inbounds R[i,j] = σ2*exp(-0.5*R[i,j]/ℓ2)
        if i != j; @inbounds R[j,i] = R[i,j]; end;
    end
    return R
end

function grad_stack(X::Matrix{Float64}, se::SEIso)
    d, nobsv = size(X)
    ℓ2 = exp(2.0 * se.ll)
    σ2 = exp(2*se.lσ)
    dxy2 = pairwise(SqEuclidean(), X)
    exp_dxy2 = exp(-dxy2/(2.0*ℓ2))
    
    stack = Array(Float64, nobsv, nobsv, 2)
    for i in 1:nobsv, j in 1:nobsv
        @inbounds stack[i,j,1] = σ2*dxy2[i,j]/ℓ2 * exp_dxy2[i,j] # dK_dℓ
        @inbounds stack[i,j,2] = 2.0 * σ2 * exp_dxy2[i,j]        # dK_dσ
    end

    return stack
end
