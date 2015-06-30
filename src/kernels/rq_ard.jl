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
    ℓ    = exp(rq.ll)
    σ2 = exp(2*rq.lσ)
    α  = exp(rq.lα)
    #K =  σ2*(1+0.5*(norm((x-y)./ℓ)^2)/α)^(-α)
    K =  σ2*(1+0.5*(wsqeuclidean(x,y,1.0./(ℓ.^2)))/α)^(-α)
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
    ℓ    = exp(rq.ll)
    σ2 = exp(2*rq.lσ)
    α  = exp(rq.lα)

    wdiff = ((x-y)./ℓ).^2
    dxy2 = sum(wdiff)
    
    dK_ℓ   = σ2*wdiff*(1+0.5*dxy2/α)^(-α-1)
    dK_σ = 2.0*σ2*(1+0.5*dxy2/α)^(-α)
    
    part     = (1+0.5*dxy2/α)
    dK_α = σ2*part^(-α)*(0.5*dxy2/part-α*log(part))
    dK_theta = [dK_ℓ,dK_σ,dK_α]
    return dK_theta
end
