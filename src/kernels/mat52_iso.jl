# Matern 5/2 isotropic covariance function

@doc """
# Description
Constructor for the isotropic Matern 5/2 kernel (covariance)

k(x,x') = σ²(1+√5*d/ℓ + 5d²/3ℓ²)exp(-√5*d/ℓ), where d = |x-x'|
# Arguments:
* `ll::Float64`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type Mat52Iso <: Kernel
    ll::Float64      # Log of Length scale 
    lσ::Float64      # Log of signal std
    Mat52Iso(ll::Float64, lσ::Float64) = new(ll, lσ)
end

function kern(mat::Mat52Iso, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(mat.ll)
    sigma2 = exp(2*mat.lσ)
    K = sigma2*(1+sqrt(5)*norm(x-y)/ell+5*norm(x-y)^2/(3*ell^2))*exp(-sqrt(5)*norm(x-y)/ell)
    return K
end


get_params(mat::Mat52Iso) = Float64[mat.ll, mat.lσ]

num_params(mat::Mat52Iso) = 2

function set_params!(mat::Mat52Iso, hyp::Vector{Float64})
    length(hyp) == 2 || throw(ArgumentError("Matern 5/2 only has two parameters"))
    mat.ll, mat.lσ = hyp
end

function grad_kern(mat::Mat52Iso, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(mat.ll)
    sigma2 = exp(2*mat.lσ)
    
    dK_ell = sigma2*((5*norm(x-y)^2/ell^2)*(1+sqrt(5)*norm(x-y)/ell)/3)*exp(-sqrt(5)*norm(x-y)/ell)
    dK_sigma = 2.0*sigma2*(1+sqrt(5)*norm(x-y)/ell+5*norm(x-y)^2/(3*ell^2))*exp(-sqrt(5)*norm(x-y)/ell)
    
    dK_theta = [dK_ell,dK_sigma]
    return dK_theta
end
