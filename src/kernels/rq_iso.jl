# Rational Quadratic Isotropic Covariance Function

"""
    RQIso <: Isotropic{SqEuclidean}

Isotropic Rational Quadratic kernel (covariance)
```math
k(x,x') = σ²(1 + (x - x')ᵀ(x - x')/(2αℓ²))^{-α}
```
with length scale ``ℓ``, signal standard deviation ``σ``, and shape parameter ``α``.
"""
mutable struct RQIso <: Isotropic{SqEuclidean}
    "Squared length scale"
    ℓ2::Float64
    "Signal variance"
    σ2::Float64
    "Shape parameter"
    α::Float64
    "Priors for kernel parameters"
    priors::Array

    """
        RQIso(ll:Float64, lσ::Float64, lα::Float64)

    Create `RQIso` with length scale `exp(ll)`, signal standard deviation `exp(lσ)`, and
    shape parameter `exp(lα)`.
    """
    RQIso(ll::Float64, lσ::Float64, lα::Float64) =
        new(exp(2 * ll), exp(2 * lσ), exp(lα), [])
end

function set_params!(rq::RQIso, hyp::VecF64)
    length(hyp) == 3 || throw(ArgumentError("Rational Quadratic function has three parameters"))
    rq.ℓ2, rq.σ2, rq.α = exp(2 * hyp[1]), exp(2 * hyp[2]), exp(hyp[3])
end

get_params(rq::RQIso) = Float64[log(rq.ℓ2) / 2, log(rq.σ2) / 2, log(rq.α)]
get_param_names(rq::RQIso) = [:ll, :lσ, :lα]
num_params(rq::RQIso) = 3

Statistics.cov(rq::RQIso, r::Number) = rq.σ2 * (1 + r / (2 * rq.α * rq.ℓ2))^(-rq.α)

@inline dk_dll(rq::RQIso, r::Float64) =
    (s = r / rq.ℓ2; rq.σ2 * s * (1 + s / (2 * rq.α))^(-rq.α - 1)) # dK_d(log ℓ)dK_dℓ
@inline function dk_dlα(rq::RQIso, r::Float64)
    s = r / rq.ℓ2
    part = 1 + s / (2 * rq.α)
    rq.σ2 * part^(-rq.α) * (s / (2 * part) - rq.α * log(part))  # dK_d(log α)
end
@inline function dk_dθp(rq::RQIso, r::Float64, p::Int)
    if p==1
        return dk_dll(rq, r)
    elseif p==2
        return dk_dlσ(rq, r)
    elseif p==3
        return dk_dlα(rq, r)
    else
        return NaN
    end
end
