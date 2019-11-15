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
mutable struct RQArd{T<:Real} <: StationaryARD{WeightedSqEuclidean}
    "Inverse squared length scale"
    iℓ2::Vector{T}
    "Signal variance"
    σ2::T
    "Shape parameter"
    α::T
    "Priors for kernel parameters"
    priors::Array
end

"""
Rational Quadratic ARD Covariance Function

    RQArd(ll::Vector{Real}, lσ::Real, lα::Real)

# Arguments
  - `ll::Vector{Real}`: vector of length scales (given on log scale)
  - `lσ::Real`: signal standard deviation (given on log scale)
  - `lα::Real`: shape parameter (given on log scale)  
"""
RQArd(ll::Vector{T}, lσ::T, lα::T) where T = RQArd{T}(exp.(-2 .* ll), exp(2 * lσ), exp(lα), [])

function set_params!(rq::RQArd, hyp::AbstractVector)
    length(hyp) == num_params(rq) || throw(ArgumentError("RQArd kernel has $(num_params(rq_ard)) parameters"))
    @views @. rq.iℓ2 = exp(-2 * hyp[1:(end-2)])
    rq.σ2 = exp(2 * hyp[end-1])
    rq.α = exp(hyp[end])
end

get_params(rq::RQArd) = [-log.(rq.iℓ2) / 2; log(rq.σ2) / 2; log(rq.α)]
get_param_names(rq::RQArd) = [get_param_names(rq.iℓ2, :ll); :lσ; :lα]
num_params(rq::RQArd) = length(rq.iℓ2) + 2

cov(rq::RQArd,r::Number) = rq.σ2*(1+0.5*r/rq.α)^(-rq.α)

@inline dk_dll(rq::RQArd, r::Real, wdiffp::Real) =
    rq.σ2 * wdiffp * (1 + r / (2 * rq.α))^(-rq.α - 1)
@inline function dk_dlα(rq::RQArd, r::Real)
    part = (1 + r / (2 * rq.α))
    return rq.σ2 * part^(-rq.α) * (r / (2 * part) - rq.α * log(part))
end
@inline function dKij_dθp(rq::RQArd, X1::AbstractMatrix, X2::AbstractMatrix, i::Int, j::Int, p::Int, dim::Int)
    if p <= dim
        return dk_dll(rq, distij(metric(rq),X1,X2,i,j,dim), distijk(metric(rq),X1,X2,i,j,p))
    elseif p==dim+1
        return dk_dlσ(rq, distij(metric(rq),X1,X2,i,j,dim))
    else
        return dk_dlα(rq, distij(metric(rq),X1,X2,i,j,dim))
    end
end
@inline function dKij_dθp(rq::RQArd, X1::AbstractMatrix, X2::AbstractMatrix, data::StationaryARDData, 
                          i::Int, j::Int, p::Int, dim::Int)
    return dKij_dθp(rq,X1,X2,i,j,p,dim)
end
