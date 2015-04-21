# Matern 5/2 ARD covariance Function

@doc """
# Description
Constructor for the ARD Matern 5/2 kernel (covariance)

k(x,x') = σ²(1+√3*d/L + 5d²/3L²)exp(-√5*d/L), where d = |x-x'| and L = diag(ℓ₁,ℓ₂,...)
# Arguments:
* `ll::Vector{Float64}`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type Mat52Ard <: Kernel
    ll::Vector{Float64}   # Log of Length scale 
    lσ::Float64           # Log of signal std
    dim::Int              # Number of hyperparameters
    Mat52Ard(ll::Vector{Float64}, lσ::Float64) = new(ll, lσ, size(ll,1)+1)
end

function kern(mat::Mat52Ard, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(mat.ll)
    sigma2 = exp(2*mat.lσ)
        K = sigma2*(1+sqrt(5)*norm((x-y)./ell)+5/3*norm((x-y)./ell)^2)*exp(-sqrt(5)*norm((x-y)./ell))
    return K
end

get_params(mat::Mat52Ard) = [mat.ll, mat.lσ]

num_params(mat::Mat52Ard) = mat.dim

function set_params!(mat::Mat52Ard, hyp::Vector{Float64})
    length(hyp) == mat.dim || throw(ArgumentError("Matern 5/2 only has $(mat.dim) parameters"))
    mat.ll = hyp[1:(mat.dim-1)]
    mat.lσ = hyp[mat.dim]
end

function grad_kern(mat::Mat52Ard, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(mat.ll)
    sigma2 = exp(2*mat.lσ)

    dK_ell = sigma2.*((5.*((x-y)./ell).^2).*(1+sqrt(5).*((x-y)./ell))/3).*exp(-sqrt(5).*norm((x-y)./ell))
    dK_sigma = 2.0*sigma2*(1+sqrt(5)*norm((x-y)./ell)+5/3*norm((x-y)./ell)^2)*exp(-sqrt(5)*norm((x-y)./ell))
    
    dK_theta = [dK_ell,dK_sigma]
    return dK_theta
end
