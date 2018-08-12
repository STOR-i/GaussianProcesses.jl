#White Noise kernel

"""
    Noise <: Kernel

Noise kernel (covariance)
```math
k(x,x') = σ²δ(x-x'),
```
where ``δ`` is the Kronecker delta function and ``σ`` is the signal standard deviation.
"""
mutable struct Noise <: Kernel
    "Signal variance"
    σ2::Float64
    "Priors for kernel parameters"
    priors::Array

    """
        Noise(lσ::Float64)

    Create `Noise` with signal standard deviation `exp(lσ)`.
    """
    Noise(lσ::Float64) = new(exp(2 * lσ), [])
end

Statistics.cov(noise::Noise, sameloc::Bool) = noise.σ2*sameloc
function Statistics.cov(noise::Noise, x::VecF64, y::VecF64)
    return cov(noise, (norm(x-y)<eps()))
end

get_params(noise::Noise) = Float64[log(noise.σ2)/2.0]
get_param_names(noise::Noise) = [:lσ]
num_params(noise::Noise) = 1

function set_params!(noise::Noise, hyp::VecF64)
    length(hyp) == 1 || throw(ArgumentError("Noise kernel only has one parameter"))
    noise.σ2 = exp(2.0*hyp[1])
end

@inline dk_dlσ(noise::Noise, sameloc=Bool) = 2.0*cov(noise,sameloc)
@inline function dKij_dθp(noise::Noise, X::MatF64, i::Int, j::Int, p::Int, dim::Int)
    return dk_dlσ(noise, norm(X[:,i]-X[:,j])<eps())
end
@inline function dKij_dθp(noise::Noise, X::MatF64, data::EmptyData, i::Int, j::Int, p::Int, dim::Int)
    return dKij_dθp(noise, X, i, j, p, dim)
end
