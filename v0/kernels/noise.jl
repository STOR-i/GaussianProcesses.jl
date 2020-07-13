# White Noise kernel

"""
    Noise <: Kernel

Noise kernel (covariance)
```math
k(x,x') = σ²δ(x-x'),
```
where ``δ`` is the Kronecker delta function and ``σ`` is the signal standard deviation.
"""
mutable struct Noise{T<:Real} <: Kernel
    "Signal variance"
    σ2::T
    "Priors for kernel parameters"
    priors::Array
end

"""
White Noise kernel
    
    Noise(lσ::Real)

# Arguments
  - `lσ::Real`: signal standard deviation (given on log scale)  
"""
Noise(lσ::T) where T = Noise{T}(exp(2 * lσ), [])

cov(noise::Noise, sameloc::Bool) = sameloc ? noise.σ2 : zero(noise.σ2)
cov(noise::Noise, x::AbstractVector, y::AbstractVector) = cov(noise, x ≈ y)
@inline @inbounds function cov_ij(noise::Noise, X1::AbstractMatrix, X2::AbstractMatrix, i::Int, j::Int, dim::Int)
    @inbounds for z in 1:dim
        if !(X1[z,i] ≈ X2[z,j])
            return zero(noise.σ2)
        end
    end
    return noise.σ2
    # return cov(noise, all(X1[z,i] ≈ X2[z,j] for z in 1:dim)) # this is inefficient for some reason
end

get_params(noise::Noise{T}) where T = T[log(noise.σ2) / 2]
get_param_names(noise::Noise) = [:lσ]
num_params(noise::Noise) = 1

function set_params!(noise::Noise, hyp::AbstractVector)
    length(hyp) == 1 || throw(ArgumentError("Noise kernel has one parameter, received $(length(hyp))."))
    noise.σ2 = exp(2 * hyp[1])
end

@inline dKij_dθp(noise::Noise, X1::AbstractMatrix, X2::AbstractMatrix, i::Int, j::Int, p::Int, dim::Int) =
    # 2 * cov(noise, view(X, :, i), view(X, :, j))
    2 * cov_ij(noise, X1, X2, i, j, dim)
