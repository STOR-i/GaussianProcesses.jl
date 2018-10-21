"""
    SEIso <: Isotropic{SqEuclidean}

Isotropic Squared Exponential kernel (covariance)
```math
k(x,x') = σ²\\exp(- (x - x')ᵀ(x - x')/(2ℓ²))
```
with length scale ``ℓ`` and signal standard deviation ``σ``.
"""
mutable struct SEIso <: Isotropic{SqEuclidean}
    "Squared length scale"
    ℓ2::Float64
    "Signal variance"
    σ2::Float64
    "Priors for kernel parameters"
    priors::Array

    """
        SEIso(ll::Float64, lσ::Float64)

    Create `SEIso` with length scale `exp(ll)` and signal standard deviation `exp(lσ)`.
    """
    SEIso(ll::Float64, lσ::Float64) = new(exp(2 * ll), exp(2 * lσ), [])
end

function set_params!(se::SEIso, hyp::VecF64)
    length(hyp) == 2 || throw(ArgumentError("Squared exponential only has two parameters"))
    se.ℓ2, se.σ2 = exp(2 * hyp[1]), exp(2 * hyp[2])
end

get_params(se::SEIso) = Float64[log(se.ℓ2) / 2, log(se.σ2) / 2]
get_param_names(se::SEIso) = [:ll, :lσ]
num_params(se::SEIso) = 2

Statistics.cov(se::SEIso, r::Number) = se.σ2*exp(-0.5*r/se.ℓ2)

@inline dk_dll(se::SEIso, r::Float64) = r/se.ℓ2*cov(se,r)
@inline function dk_dθp(se::SEIso, r::Float64, p::Int)
    if p==1
        return dk_dll(se, r)
    elseif p==2
        return dk_dlσ(se, r)
    else
        return NaN
    end
end
