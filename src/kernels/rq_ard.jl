# Rational Quadratic ARD Covariance Function

"""
    RQArd <: StationaryARD{WeightedSqEuclidean}

ARD Rational Quadratic kernel (covariance)
```math
k(x,x') = σ²(1 + (x - x')ᵀL⁻²(x - x')/(2α))^{-α}
```
with length scale ``ℓ = (ℓ₁, ℓ₂, …)``, signal standard deviation ``σ``, and shape parameter
``α`` where ``L = diag(ℓ₁, ℓ₂, …)``.
"""
mutable struct RQArd <: StationaryARD{WeightedSqEuclidean}
    "Inverse squared length scale"
    iℓ2::Vector{Float64}
    "Signal variance"
    σ2::Float64
    "Shape parameter"
    α::Float64
    "Priors for kernel parameters"
    priors::Array

    """
        RQArd(ll::Vector{Float64}, lσ::Float64, lα::Float64)

    Create `RQArd` with length scale `exp.(ll)`, signal standard deviation `exp(lσ)`, and
    shape parameter `exp(lα)`.
    """
    RQArd(ll::Vector{Float64}, lσ::Float64, lα::Float64) =
        new(exp.(-2 .* ll), exp(2 * lσ), exp(lα), [])
end

function set_params!(rq::RQArd, hyp::VecF64)
    length(hyp) == num_params(rq) || throw(ArgumentError("RQArd kernel has $(num_params(rq_ard)) parameters"))
    @views @. rq.iℓ2 = exp(-2 * hyp[1:(end-2)])
    rq.σ2 = exp(2 * hyp[end-1])
    rq.α = exp(hyp[end])
end

get_params(rq::RQArd) = [-log.(rq.iℓ2) / 2; log(rq.σ2) / 2; log(rq.α)]
get_param_names(rq::RQArd) = [get_param_names(rq.iℓ2, :ll); :lσ; :lα]
num_params(rq::RQArd) = length(rq.iℓ2) + 2

Statistics.cov(rq::RQArd,r::Number) = rq.σ2*(1+0.5*r/rq.α)^(-rq.α)

@inline dk_dll(rq::RQArd, r::Float64, wdiffp::Float64) =
    rq.σ2 * wdiffp * (1 + r / (2 * rq.α))^(-rq.α - 1)
@inline function dk_dlα(rq::RQArd, r::Float64)
    part = (1 + r / (2 * rq.α))
    return rq.σ2 * part^(-rq.α) * (r / (2 * part) - rq.α * log(part))
end
@inline function dKij_dθp(rq::RQArd, X::MatF64, i::Int, j::Int, p::Int, dim::Int)
    if p <= dim
        return dk_dll(rq, distij(metric(rq),X,i,j,dim), distijk(metric(rq),X,i,j,p))
    elseif p==dim+1
        return dk_dlσ(rq, distij(metric(rq),X,i,j,dim))
    else
        return dk_dlα(rq, distij(metric(rq),X,i,j,dim))
    end
end
@inline function dKij_dθp(rq::RQArd, X::MatF64, data::StationaryARDData, i::Int, j::Int, p::Int, dim::Int)
    return dKij_dθp(rq,X,i,j,p,dim)
end
