# Matern 1/2 isotropic covariance Function

@doc """
# Description
Constructor for the isotropic Matern 1/2 kernel (covariance)

k(x,x') = σ²exp(-d/ℓ), where d=|x-x'|
# Arguments:
* `ll::Float64`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type Mat12Iso <: Kernel
    ll::Float64     #Log of length scale
    lσ::Float64     #Log of signal std
    Mat12Iso(ll::Float64, lσ::Float64) = new(ll,lσ)
end

function kern(mat::Mat12Iso, x::Vector{Float64},y::Vector{Float64})
    ℓ = exp(mat.ll)
    σ2 = exp(2*mat.lσ)
    K = σ2*exp(-euclidean(x,y)/ℓ)
    return K
end

get_params(mat::Mat12Iso) = exp(Float64[mat.ll, mat.lσ])

num_params(mat::Mat12Iso) = 2

function set_params!(mat::Mat12Iso, hyp::Vector{Float64})
    length(hyp) == 2 || throw(ArgumentError("Matern 1/2 covariance function only has two parameters"))
    mat.ll, mat.lσ = hyp
end

function grad_kern(mat::Mat12Iso, x::Vector{Float64}, y::Vector{Float64})
    ℓ = exp(mat.ll)
    σ2 = exp(2*mat.lσ)
    dxy = euclidean(x,y)
    exp_dxy = exp(-dxy/ℓ)
    
    dK_dℓ = σ2*dxy/ℓ*exp_dxy
    dK_dσ = 2.0*σ2*exp_dxy
    dK_dθ = [dK_dℓ,dK_dσ]
    
    return dK_dθ
end

function crossKern(X::Matrix{Float64}, k::Mat12Iso)
    ℓ = exp(k.ll)
    σ2 = exp(2*k.lσ)
    R = pairwise(Euclidean(), X)
    broadcast!(/, R, R, -ℓ)
    map!(exp, R, R)
    broadcast!(*, R, R, σ2)
    return R
end

function grad_stack!(stack::AbstractArray, X::Matrix{Float64}, mat::Mat12Iso)
    d, nobsv = size(X) 
    ℓ = exp(mat.ll)
    σ2 = exp(2*mat.lσ)

    dxy = pairwise(Euclidean(), X)
    exp_dxy = exp(-dxy/ℓ)

    for i in 1:nobsv, j in 1:nobsv
        @inbounds stack[i,j,1] = σ2*dxy[i,j]/ℓ*exp_dxy[i,j] # dK_dℓ
        @inbounds stack[i,j,2] = 2.0 * σ2 * exp_dxy[i,j]    # dK_dσ
    end
    
    return stack
end
