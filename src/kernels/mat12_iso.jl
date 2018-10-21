# Matern 1/2 isotropic covariance Function

"""
    Mat12Iso <: MaternISO

Isotropic Matern 1/2 kernel (covariance)
```math
k(x,x') = σ^2 \\exp(-|x-y|/ℓ)
```
with length scale ``ℓ`` and signal standard deviation ``σ``.
"""
mutable struct Mat12Iso <: MaternIso
    "Length scale"
    ℓ::Float64
    "Signal variance"
    σ2::Float64
    "Priors for kernel parameters"
    priors::Array

    """
        Mat12Iso(ll::Float64, lσ::Float64)

    Create `Mat12Iso` with length scale `exp(ll)` and signal standard deviation `exp(σ)`.
    """
    Mat12Iso(ll::Float64, lσ::Float64) = new(exp(ll), exp(2 * lσ), [])
end

function set_params!(mat::Mat12Iso, hyp::VecF64)
    length(hyp) == 2 || throw(ArgumentError("Matern 1/2 covariance function only has two parameters"))
    mat.ℓ, mat.σ2 = exp(hyp[1]), exp(2 * hyp[2])
end

get_params(mat::Mat12Iso) = Float64[log(mat.ℓ), log(mat.σ2) / 2]
get_param_names(mat::Mat12Iso) = [:ll, :lσ]
num_params(mat::Mat12Iso) = 2

Statistics.cov(mat::Mat12Iso, r::Number) = mat.σ2 * exp(-r / mat.ℓ)

@inline dk_dll(mat::Mat12Iso, r::Float64) = r / mat.ℓ * cov(mat, r)
