# Matern 3/2 isotropic covariance function

@doc """
# Description
Constructor for the isotropic Matern 3/2 kernel (covariance)

k(x,x') = σ²(1+√3*d/ℓ)exp(-√3*d/ℓ), where d = |x-x'|
# Arguments:
* `ll::Float64`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type Mat32Iso <: Kernel
    ll::Float64      # Log of Length scale 
    lσ::Float64      # Log of Signal std
    Mat32Iso(ll::Float64, lσ::Float64) = new(ll,lσ)
end

function kern(mat::Mat32Iso, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(mat.ll)
    sigma2 = exp(2*mat.lσ)
    K = sigma2*(1+sqrt(3)*norm(x-y)/ell)*exp(-sqrt(3)*norm(x-y)/ell)
    return K
end

get_params(mat::Mat32Iso) = Float64[mat.ll, mat.lσ]

num_params(mat::Mat32Iso) = 2

function set_params!(mat::Mat32Iso, hyp::Vector{Float64})
    length(hyp) == 2 || throw(ArgumentError("Matern 3/2 only has two parameters"))
    mat.ll, mat.lσ = hyp
end

function grad_kern(mat::Mat32Iso, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(mat.ll)
    sigma2 = exp(2*mat.lσ)
    
    dK_ell = sigma2*(sqrt(3)*norm(x-y)/ell)^2*exp(-sqrt(3)*norm(x-y)/ell)
    dK_sigma = 2.0*sigma2*(1+sqrt(3)*norm(x-y)/ell)*exp(-sqrt(3)*norm(x-y)/ell)
    dK_theta = [dK_ell,dK_sigma]
    
    return dK_theta
end    


