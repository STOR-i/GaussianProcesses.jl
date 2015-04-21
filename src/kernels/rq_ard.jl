# Rational Quadratic ARD Covariance Function 

@doc """
# Description
Constructor for the ARD Rational Quadratic kernel (covariance)

k(x,x') = σ²(1+(x-x')ᵀL⁻²(x-x')/2α)^{-α}, where L = diag(ℓ₁,ℓ₂,...)
# Arguments:
* `ll::Vector{Float64}`: Log of length scale ℓ
* `lσ::Float64`        : Log of the signal standard deviation σ
* `lα::Float64`        : Log of shape parameter α
""" ->
type RQArd <: Kernel
    ll::Vector{Float64}      # Log of length scale 
    lσ::Float64              # Log of signal std
    lα::Float64              # Log of shape parameter
    dim::Int                 # Number of hyperparameters
    RQArd(ll::Vector{Float64}, lσ::Float64, lα::Float64) = new(ll, lσ, lα, size(ll,1)+2)
end

function kern(rq::RQArd, x::Vector{Float64}, y::Vector{Float64})
    ell    = exp(rq.ll)
    sigma2 = exp(2*rq.lσ)
    alpha  = exp(rq.lα)

    K =  sigma2*(1+0.5*(norm((x-y)./ell)^2)/alpha)^(-alpha)
    return K
end

get_params(rq::RQArd) = [rq.ll, rq.lσ, rq.lα]
num_params(rq::RQArd) = rq.dim

function set_params!(rq::RQArd, hyp::Vector{Float64})
    length(hyp) == rq.dim || throw(ArgumentError("Rational Quadratic ARD function has $(rq.dim) parameters"))
    rq.ll = hyp[1:rq.dim-2]
    rq.lσ = hyp[rq.dim-1]
    rq.lα = hyp[rq.dim]
end

function grad_kern(rq::RQArd, x::Vector{Float64}, y::Vector{Float64})
    ell    = exp(rq.ll)
    sigma2 = exp(2*rq.lσ)
    alpha  = exp(rq.lα)
    
    dK_ell   = sigma2*(((x-y)./ell).^2)*(1+0.5*(norm((x-y)./ell)^2)/alpha)^(-alpha-1)
    dK_sigma = 2.0*sigma2*(1+0.5*(norm((x-y)./ell)^2)/alpha)^(-alpha)
    
    part     = (1+0.5*(norm((x-y)./ell)^2)/alpha)
    dK_alpha = sigma2*part^(-alpha)*(0.5*(norm((x-y)./ell)^2)/part-alpha*log(part))
    dK_theta = [dK_ell,dK_sigma,dK_alpha]
    return dK_theta
end
