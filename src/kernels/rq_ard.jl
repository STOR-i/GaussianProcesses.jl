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
type RQArd <: Stationary
    ℓ2::Vector{Float64}      # Log of length scale 
    σ2::Float64              # Log of signal std
    α::Float64              # Log of shape parameter
    dim::Int                 # Number of hyperparameters
    RQArd(ll::Vector{Float64}, lσ::Float64, lα::Float64) = new(exp(2.0*ll), exp(2.0*lσ), exp(lα), size(ll,1)+2)
end

function set_params!(rq::RQArd, hyp::Vector{Float64})
    length(hyp) == rq.dim || throw(ArgumentError("Rational Quadratic ARD function has $(rq.dim) parameters"))
    rq.ℓ2 = exp(2.0*hyp[1:rq.dim-2])
    rq.σ2 = exp(2.0*hyp[rq.dim-1])
    rq.α = exp(hyp[rq.dim])
end

get_params(rq::RQArd) = [log(rq.ℓ2)/2.0; log(rq.σ2)/2.0; log(rq.α)]
num_params(rq::RQArd) = rq.dim

metric(rq::RQArd) = WeightedSqEuclidean(1.0./(rq.ℓ2))
kern(rq::RQArd,r::Float64) = rq.σ2*(1+0.5*r/rq.α)^(-rq.α)
    

function grad_kern(rq::RQArd, x::Vector{Float64}, y::Vector{Float64})
    r = distance(rq, x, y)

    wdiff = ((x-y).^2)./rq.ℓ2
    dxy2 = sum(wdiff)
    
    g1   = rq.σ2*wdiff*(1+0.5*dxy2/rq.α)^(-rq.α-1)
    g2 = 2.0*rq.σ2*(1+0.5*dxy2/rq.α)^(-rq.α)
    
    part     = (1+0.5*dxy2/rq.α)
    g3 = rq.σ2*part^(-rq.α)*(0.5*dxy2/part-rq.α*log(part))
    return [g1; g2; g3]
end
