# Squared Exponential Function with ARD

@doc """
# Description
Constructor for the ARD Squared Exponential kernel (covariance)

k(x,x') = σ²exp(-(x-x')ᵀL⁻²(x-x')/2), where L = diag(ℓ₁,ℓ₂,...)
# Arguments:
* `ll::Vector{Float64}`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type SEArd <: Kernel
    ll::Vector{Float64}      # Log of Length scale
    lσ::Float64              # Log of Signal std
    dim::Int                 # Number of hyperparameters
    SEArd(ll::Vector{Float64}, lσ::Float64) = new(ll,lσ, size(ll,1)+1)
end

function kern(se::SEArd, x::Vector{Float64}, y::Vector{Float64})
    ℓ    = exp(se.ll)
    σ2 = exp(2*se.lσ)
    K = σ2*exp(-0.5*wsqeuclidean(x,y,1.0./(ℓ.^2)))
    return K
end

get_params(se::SEArd) = [se.ll, se.lσ]
num_params(se::SEArd) = se.dim

function set_params!(se::SEArd, hyp::Vector{Float64})
    length(hyp) == se.dim || throw(ArgumentError("Squared exponential ARD only has $(se.dim) parameters"))
    se.ll = hyp[1:(se.dim-1)]
    se.lσ = hyp[se.dim]
end

function grad_kern(se::SEArd, x::Vector{Float64}, y::Vector{Float64})
    ℓ = exp(se.ll)
    σ2 = exp(2*se.lσ)

    wdiff = ((x-y)./ℓ).^2
    dxy2 = sum(wdiff)
    
    dK_dℓ   = σ2.*wdiff*exp(-0.5*dxy2)
    dK_dσ = 2.0*σ2*exp(-0.5*dxy2)
    
    dK_theta = [dK_dℓ,dK_dσ]
    return dK_theta
end
