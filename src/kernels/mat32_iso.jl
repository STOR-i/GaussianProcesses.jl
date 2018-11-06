# Matern 3/2 isotropic covariance function

"""
    Mat32Iso <: MaternIso

Isotropic Matern 3/2 kernel (covariance)
```math
k(x,x') = σ²(1 + √3|x-x'|/ℓ)\\exp(-√3|x-x'|/ℓ)
```
with length scale ``ℓ`` and signal standard deviation ``σ``.
"""
mutable struct Mat32Iso{T<:Real} <: MaternIso
    "Length scale"
    ℓ::T
    "Signal variance"
    σ2::T
    "Priors for kernel parameters"
    priors::Array
end

"""
    Mat32Iso(ll::T, lσ::T)

Create `Mat32Iso` with length scale `exp(ll)` and signal standard deviation `exp(lσ)`.
"""
Mat32Iso(ll::T, lσ::T) where T = Mat32Iso{T}(exp(ll), exp(2 * lσ), [])

function set_params!(mat::Mat32Iso, hyp::AbstractVector)
    length(hyp) == 2 || throw(ArgumentError("Matern 3/2 has two parameters, received $(length(hyp))."))
    mat.ℓ, mat.σ2 = exp(hyp[1]), exp(2 * hyp[2])
end

get_params(mat::Mat32Iso{T}) where T = T[log(mat.ℓ), log(mat.σ2) / 2 ]
get_param_names(mat::Mat32Iso) = [:ll, :lσ]
num_params(mat::Mat32Iso) = 2

cov(mat::Mat32Iso, r::Number) =
    (s = √3 * r / mat.ℓ; mat.σ2 * (1 + s) * exp(-s))

@inline dk_dll(mat::Mat32Iso, r::Real) =
    (s = √3 * r / mat.ℓ; mat.σ2 * s^2 * exp(-s))
