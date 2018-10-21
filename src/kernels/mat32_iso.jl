# Matern 3/2 isotropic covariance function

"""
    Mat32Iso <: MaternIso

Isotropic Matern 3/2 kernel (covariance)
```math
k(x,x') = σ²(1 + √3|x-x'|/ℓ)\\exp(-√3|x-x'|/ℓ)
```
with length scale ``ℓ`` and signal standard deviation ``σ``.
"""
mutable struct Mat32Iso <: MaternIso
    "Length scale"
    ℓ::Float64
    "Signal variance"
    σ2::Float64
    "Priors for kernel parameters"
    priors::Array

    """
        Mat32Iso(ll::Float64, lσ::Float64)

    Create `Mat32Iso` with length scale `exp(ll)` and signal standard deviation `exp(lσ)`.
    """
    Mat32Iso(ll::Float64, lσ::Float64) = new(exp(ll), exp(2 * lσ), [])
end

function set_params!(mat::Mat32Iso, hyp::VecF64)
    length(hyp) == 2 || throw(ArgumentError("Matern 3/2 only has two parameters"))
    mat.ℓ, mat.σ2 = exp(hyp[1]), exp(2 * hyp[2])
end

get_params(mat::Mat32Iso) = Float64[log(mat.ℓ), log(mat.σ2) / 2 ]
get_param_names(mat::Mat32Iso) = [:ll, :lσ]
num_params(mat::Mat32Iso) = 2

Statistics.cov(mat::Mat32Iso, r::Number) =
    (s = √3 * r / mat.ℓ; mat.σ2 * (1 + s) * exp(-s))

@inline dk_dll(mat::Mat32Iso, r::Float64) =
    (s = √3 * r / mat.ℓ; mat.σ2 * s^2 * exp(-s))
