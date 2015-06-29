# Matern 1/2 ARD covariance Function

@doc """
# Description
Constructor for the ARD Matern 1/2 kernel (covariance)

k(x,x') = σ²exp(-d/L), where d = |x-x'| and L = diag(ℓ₁,ℓ₂,...)
# Arguments:
* `ll::Vector{Float64}`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type Mat12Ard <: Kernel
    ll::Vector{Float64}     # Log of length scale
    lσ::Float64             # Log of signal std
    dim::Int                # Number of hyperparameters
    Mat12Ard(ll::Vector{Float64}, lσ::Float64) = new(ll,lσ, size(ll,1)+1)
end

function kern(mat::Mat12Ard, x::Vector{Float64},y::Vector{Float64})
    ell = exp(mat.ll)
    sigma2 = exp(2*mat.lσ)
    K = sigma2*exp(-weuclidean(x,y, 1.0./(ell.^2)))
    return K
end

get_params(mat::Mat12Ard) = [mat.ll, mat.lσ]

num_params(mat::Mat12Ard) = mat.dim

function set_params!(mat::Mat12Ard, hyp::Vector{Float64})
    length(hyp) == mat.dim || throw(ArgumentError("Matern 1/2 covariance function only has $(mat.dim) parameters"))
    mat.ll = hyp[1:(mat.dim-1)]
    mat.lσ = hyp[mat.dim]
end

function grad_kern(mat::Mat12Ard, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(mat.ll)
    sigma2 = exp(2*mat.lσ)
    wdiff = (x-y)./ell
    dxy = norm(wdiff)
    
    dK_ell = sigma2*wdiff*exp(-dxy)
    dK_sigma = 2.0*sigma2*exp(-dxy)
    dK_theta = [dK_ell,dK_sigma]
    
    return dK_theta
end
