# Matern 1/2 isotropic covariance Function

"""
    Mat12Iso <: MaternISO

Isotropic Matern 1/2 kernel (covariance)
```math
k(x,x') = σ^2 \\exp(-|x-y|/ℓ)
```
with length scale ``ℓ`` and signal standard deviation ``σ``.
"""
mutable struct Mat12Iso{T} <: MaternIso where {T<:Real}
    "Length scale"
    ℓ::T
    "Signal variance"
    σ2::T
    "Priors for kernel parameters"
    priors::Array
end

"""
    Mat12Iso(ll::T, lσ::T)

Create `Mat12Iso` with length scale `exp(ll)` and signal standard deviation `exp(σ)`.
"""
Mat12Iso(ll::T, lσ::T) where T = Mat12Iso{T}(exp(ll), exp(2 * lσ), [])

function set_params!(mat::Mat12Iso, hyp::AbstractVector)
    length(hyp) == 2 || throw(ArgumentError("Matern 1/2 covariance function has two parameters, received $(length(hyp))."))
    mat.ℓ, mat.σ2 = exp(hyp[1]), exp(2 * hyp[2])
end

get_params(mat::Mat12Iso{T}) where T = T[log(mat.ℓ), log(mat.σ2) / 2]
get_param_names(mat::Mat12Iso) = [:ll, :lσ]
num_params(mat::Mat12Iso) = 2

cov(mat::Mat12Iso, r::Number) = mat.σ2 * exp(-r / mat.ℓ)

@inline dk_dll(mat::Mat12Iso, r::Real) = r / mat.ℓ * cov(mat, r)
