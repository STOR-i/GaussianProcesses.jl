# Squared Exponential Function with ARD

"""
    SEArd <: StationaryARD{WeightedSqEuclidean}

ARD Squared Exponential kernel (covariance)
```math
k(x,x') = σ²\\exp(- (x - x')ᵀL⁻²(x - y)/2)
```
with length scale ``ℓ = (ℓ₁, ℓ₂, …)`` and signal standard deviation ``σ`` where
``L = diag(ℓ₁, ℓ₂, …)``.
"""
mutable struct SEArd <: StationaryARD{WeightedSqEuclidean}
    "Inverse squared length scale"
    iℓ2::Vector{Float64}
    "Signal variance"
    σ2::Float64
    "Priors for kernel parameters"
    priors::Array

    """
        SEArd(ll::Vector{Float64}, lσ::Float64)

    Create `SEArd` with length scale `exp.(ll)` and signal standard deviation `exp(lσ)`.
    """
    SEArd(ll::Vector{Float64}, lσ::Float64) = new(exp.(-2 .* ll), exp(2 * lσ), [])
end

function set_params!(se::SEArd, hyp::VecF64)
    length(hyp) == num_params(se) || throw(ArgumentError("SEArd only has $(num_params(se)) parameters"))
    @views @. se.iℓ2 = exp(-2 * hyp[1:(end-1)])
    se.σ2 = exp(2 * hyp[end])
end

get_params(se::SEArd) = [-log.(se.iℓ2) / 2 ; log(se.σ2) / 2]
get_param_names(k::SEArd) = [get_param_names(k.iℓ2, :ll); :lσ]
num_params(se::SEArd) = length(se.iℓ2) + 1

Statistics.cov(se::SEArd, r::Number) = se.σ2*exp(-r / 2)

@inline dk_dll(se::SEArd, r::Float64, wdiffp::Float64) = wdiffp*cov(se,r)
@inline function dKij_dθp(se::SEArd, X::MatF64, i::Int, j::Int, p::Int, dim::Int)
    if p <= dim
        return dk_dll(se, distij(metric(se),X,i,j,dim), distijk(metric(se),X,i,j,p))
    elseif p==dim+1
        return dk_dlσ(se, distij(metric(se),X,i,j,dim))
    else
        return NaN
    end
end
@inline function dKij_dθp(se::SEArd, X::MatF64, data::StationaryARDData, i::Int, j::Int, p::Int, dim::Int)
    return dKij_dθp(se,X,i,j,p,dim)
end
