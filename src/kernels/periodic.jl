# Periodic Function

"""
    Periodic <: Isotropic{Euclidean}

Periodic kernel (covariance)
```math
k(x,x') = σ²\\exp(-2\\sin²(π|x-x'|/p)/ℓ²)
```
with length scale ``ℓ``, signal standard deviation ``σ``, and period ``p``.
"""
mutable struct Periodic{T<:Real} <: Isotropic{Euclidean}
    "Squared length scale"
    ℓ2::T
    "Signal variance"
    σ2::T
    "Period"
    p::T
    "Priors for kernel parameters"
    priors::Array
end

"""
    Periodic(ll::Real, lσ::Real, lp::Real)

Create `Periodic` with length scale `exp(ll)`, signal standard deviation `exp(lσ)`, and
period `exp(lp)`.
"""
Periodic(ll::T, lσ::T, lp::T) where T = Periodic{T}(exp(2 * ll), exp(2 * lσ), exp(lp), [])

get_params(pe::Periodic{T}) where T = T[log(pe.ℓ2) / 2, log(pe.σ2) / 2, log(pe.p)]
get_param_names(pe::Periodic) = [:ll, :lσ, :lp]
num_params(pe::Periodic) = 3

function set_params!(pe::Periodic, hyp::AbstractVector)
    length(hyp) == 3 || throw(ArgumentError("Periodic function has three parameters, received $(length(hyp))."))
    pe.ℓ2, pe.σ2 = exp(2 * hyp[1]), exp(2 * hyp[2])
    pe.p = exp(hyp[3])
end

cov(pe::Periodic, r::Number) = pe.σ2 * exp(-2 / pe.ℓ2 * sin(π * r / pe.p)^2)


@inline dk_dll(pe::Periodic, r::Real) =
    (s = 2 * sin(π * r / pe.p)^2 / pe.ℓ2; 2 * pe.σ2 * s * exp(-s)) # dK_dlogℓ
@inline dk_dlp(pe::Periodic, r::Real) =
    (s = π * r / pe.p; t = 2 / pe.ℓ2; pe.σ2 * s * t * sin(2 * s) * exp(-t * sin(s)^2))  # dK_dlogp

@inline function dk_dθp(pe::Periodic, r::Real, p::Int)
    if p==1
        dk_dll(pe, r)
    elseif p==2
        dk_dlσ(pe, r)
    elseif p==3
        dk_dlp(pe, r)
    else
        return NaN
    end
end
