# Matern 3/2 ARD covariance function

@doc """
# Description
Constructor for the ARD Matern 3/2 kernel (covariance)

k(x,x') = σ²(1+√3*d/L)exp(-√3*d/L), where d = |x-x'| and L = diag(ℓ₁,ℓ₂,...)
# Arguments:
* `ll::Vector{Float64}`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type Mat32Ard <: Kernel
    ll::Vector{Float64}    # Log of Length scale 
    lσ::Float64            # Log of Signal std
    dim::Int               # Number of hyperparameters
    Mat32Ard(ll::Vector{Float64}, lσ::Float64) = new(ll, lσ, size(ll,1)+1)
end

function kern(mat::Mat32Ard, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(mat.ll)
    sigma2 = exp(2*mat.lσ)

    K = sigma2*(1+sqrt(3)*norm((x-y)./ell))*exp(-sqrt(3)*norm((x-y)./ell))

    return K
end

get_params(mat::Mat32Ard) = [mat.ll, mat.lσ]

num_params(mat::Mat32Ard) = mat.dim

function set_params!(mat::Mat32Ard, hyp::Vector{Float64})
    length(hyp) == mat.dim || throw(ArgumentError("Matern 3/2 only has $(mat.dim) parameters"))
    mat.ll = hyp[1:(mat.dim-1)]
    mat.lσ = hyp[mat.dim]
end

function grad_kern(mat::Mat32Ard, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(mat.ll)
    sigma2 = exp(2*mat.lσ)

    dK_ell = sigma2.*(sqrt(3).*((x-y)./ell).^2).*exp(-sqrt(3).*norm((x-y)./ell))
    dK_sigma = 2.0*sigma2*(1+sqrt(3)*norm((x-y)./ell))*exp(-sqrt(3)*norm((x-y)./ell))
    dK_theta = [dK_ell,dK_sigma]
    
    return dK_theta
end    
