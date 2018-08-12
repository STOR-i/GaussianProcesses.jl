# Matern 5/2 isotropic covariance function

"""
    Mat52Iso <: MaternIso

Isotropic Matern 5/2 kernel (covariance)
```math
k(x,x') = σ²(1+√5|x-x'|/ℓ + 5|x-x'|²/(3ℓ²))\\exp(- √5|x-x'|/ℓ)
```
with length scale ``ℓ`` and signal standard deviation ``σ``.
"""
mutable struct Mat52Iso <: MaternIso
    "Length scale"
    ℓ::Float64
    "Signal variance"
    σ2::Float64
    "Priors for kernel parameters"
    priors::Array

    """
        Mat52Iso(ll::Float64, lσ::Float64)

    Create `Mat52Iso` with length scale `exp(ll)` and signal standard deviation `exp(lσ)`.
    """
    Mat52Iso(ll::Float64, lσ::Float64) = new(exp(ll), exp(2 * lσ), [])
end

function set_params!(mat::Mat52Iso, hyp::VecF64)
    length(hyp) == 2 || throw(ArgumentError("Matern 5/2 only has two parameters"))
    mat.ℓ, mat.σ2 = exp(hyp[1]), exp(2.0*hyp[2])
end
get_params(mat::Mat52Iso) = Float64[log(mat.ℓ), log(mat.σ2)/2.0]
get_param_names(mat::Mat52Iso) = [:ll, :lσ]
num_params(mat::Mat52Iso) = 2

Statistics.cov(mat::Mat52Iso, r::Float64) = mat.σ2*(1+sqrt(5)*r/mat.ℓ+5*r^2/(3*mat.ℓ^2))*exp(-sqrt(5)*r/mat.ℓ)

@inline dk_dll(mat::Mat52Iso, r::Float64) = mat.σ2*(5*r^2/mat.ℓ^2)*((1+sqrt(5)*r/mat.ℓ)/3)*exp(-sqrt(5)*r/mat.ℓ)
