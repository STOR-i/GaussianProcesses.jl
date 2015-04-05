# Squared Exponential Function with ARD

@doc """
# Description
This type defines the ARD squared exponential kernel (covariance), which is a positive definite function of two points in the input space
# Arguments:
 `ll::Vector{Float64}`: Log of the length scale
 `lσ::Float64`: Log of the signal standard deviation
 `dim::Int`: Number of hyperparameters

# Returns:
 `(ll, lσ)::(Vector{Float64}, Float64)`: returns the kernel hyperparameters
""" ->

type SEard <: Kernel
    ll::Vector{Float64}      # Log of Length scale
    lσ::Float64              # Log of Signal std
    dim::Int                 # Number of hyperparameters
    SEard(ll::Vector{Float64}, lσ::Float64=0.0) = new(ll,lσ, size(ll,1)+1)
end

function kern(seArd::SEard, x::Vector{Float64}, y::Vector{Float64})
    ell    = exp(seArd.ll)
    sigma2 = exp(2*seArd.lσ)
    K      = sigma2*exp(-0.5*norm((x-y)./ell)^2)
    return K
end

get_params(seArd::SEard) = [seArd.ll, seArd.lσ]
num_params(seArd::SEard) = seArd.dim

function set_params!(seArd::SEard, hyp::Vector{Float64})
    length(hyp) == seArd.dim || throw(ArgumentError("Squared exponential ARD only has $(seArd.dim) parameters"))
    seArd.ll = hyp[1:(seArd.dim-1)]
    seArd.lσ = hyp[seArd.dim]
end

function grad_kern(seArd::SEard, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(seArd.ll)
    sigma2 = exp(2*seArd.lσ)
    
    dK_ell   = sigma2.*(((x-y)./ell).^2).*exp(-0.5*norm((x-y)./ell)^2)
    dK_sigma = 2.0*sigma2*exp(-0.5*norm((x-y)./ell)^2)
    
    dK_theta = [dK_ell,dK_sigma]
    return dK_theta
end
