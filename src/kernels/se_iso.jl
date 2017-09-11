"""
# Description
Constructor for the isotropic Squared Exponential kernel (covariance)

k(x,x') = σ²exp(-(x-x')ᵀ(x-x')/2ℓ²)
# Arguments:
* `ll::Float64`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
"""
@compat struct SEIso <: GaussianProcesses.Isotropic
    params::@NT(ℓ2,σ2){PositiveGPParam,PositiveGPParam}
    priors::Array       # could also be a named tuple
    SEIso(ℓ2::Float64, σ2::Float64) = new(
        # https://github.com/blackrock/NamedTuples.jl/issues/38
        @NT(ℓ2,σ2){PositiveGPParam,PositiveGPParam}(
         PositiveGPParam(ℓ2),
         PositiveGPParam(σ2)),
        [])
end

@inline metric(se::SEIso) = SqEuclidean()
@inline cov(se::SEIso, r::Float64) = get_value(se.params.σ2)*exp(-0.5*r/get_value(se.params.ℓ2))
@inline dk_dlθ(se::SEIso, r::Float64, θ::Type{Val{:ℓ2}}) = r/get_value(se.params.ℓ2)*cov(se,r)
