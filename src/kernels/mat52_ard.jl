# Matern 5/2 ARD covariance Function

"""
    Mat52Ard <: MaternARD

ARD Matern 5/2 kernel (covariance)
```math
k(x,x') = σ²(1 + √5|x-x'|/L + 5|x-x'|²/(3L²))\\exp(- √5|x-x'|/L)
```
with length scale ``ℓ = (ℓ₁, ℓ₂, …)`` and signal standard deviation ``σ`` where
``L = diag(ℓ₁, ℓ₂, …)``.
"""
mutable struct Mat52Ard <: MaternARD
    "Inverse squared length scale"
    iℓ2::VecF64
    "Signal variance"
    σ2::Float64
    "Priors for kernel parameters"
    priors::Array

    """
        Mat52Ard(ll::Vector{Float64}, lσ::Float64)

    Create `Mat52Ard` with length scale `exp.(ll)` and signal standard deviation `exp(lσ)`.
    """
    Mat52Ard(ll::VecF64, lσ::Float64) = new(exp.(-2 * ll), exp(2 * lσ), [])
end

function set_params!(mat::Mat52Ard, hyp::VecF64)
    length(hyp) == num_params(mat) || throw(ArgumentError("Mat52 kernel only has $(num_params(mat)) parameters"))
    d = length(mat.iℓ2)
    mat.iℓ2 = exp.(-2.0*hyp[1:d])
    mat.σ2 = exp(2.0*hyp[d+1])
end

get_params(mat::Mat52Ard) = [-log.(mat.iℓ2)/2.0; log(mat.σ2)/2.0]
get_param_names(mat::Mat52Ard) = [get_param_names(mat.iℓ2, :ll); :lσ]
num_params(mat::Mat52Ard) = length(mat.iℓ2) + 1

Statistics.cov(mat::Mat52Ard, r::Float64) = mat.σ2*(1+sqrt(5)*r+5/3*r^2)*exp(-sqrt(5)*r)

dk_dll(mat::Mat52Ard, r::Float64, wdiffp::Float64) = mat.σ2*(5/3)*(1+sqrt(5)*r)*exp(-sqrt(5)*r).*wdiffp
