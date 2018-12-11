"""
    SEIso <: Isotropic{SqEuclidean}

Isotropic Squared Exponential kernel (covariance)
```math
k(x,x') = σ²\\exp(- (x - x')ᵀ(x - x')/(2ℓ²))
```
with length scale ``ℓ`` and signal standard deviation ``σ``.
"""
mutable struct SEIso{T<:Real} <: Isotropic{SqEuclidean}
    "Squared length scale"
    ℓ2::T
    "Signal variance"
    σ2::T
    "Priors for kernel parameters"
    priors::Array
end

"""
    SEIso(ll::T, lσ::T)

Create `SEIso` with length scale `exp(ll)` and signal standard deviation `exp(lσ)`.
"""
SEIso(ll::T, lσ::T) where T = SEIso{T}(exp(2 * ll), exp(2 * lσ), [])

function set_params!(se::SEIso, hyp::AbstractVector)
    length(hyp) == 2 || throw(ArgumentError("Squared exponential has two parameters, received $(length(hyp))."))
    se.ℓ2, se.σ2 = exp(2 * hyp[1]), exp(2 * hyp[2])
end

get_params(se::SEIso{T}) where T = T[log(se.ℓ2) / 2, log(se.σ2) / 2]
get_param_names(se::SEIso) = [:ll, :lσ]
num_params(se::SEIso) = 2

cov(se::SEIso, r::Number) = se.σ2*exp(-0.5*r/se.ℓ2)

@inline dk_dll(se::SEIso, r::Real) = r/se.ℓ2*cov(se,r)
@inline function dk_dθp(se::SEIso, r::Real, p::Int)
    if p==1
        return dk_dll(se, r)
    elseif p==2
        return dk_dlσ(se, r)
    else
        return NaN
    end
end
