#White Noise kernel

@doc """
# Description
Constructor for the Noise kernel (covariance)

k(x,x') = σ²δ(x-x'), where δ is a Kronecker delta function and equals 1 iff x=x' and zero otherwise
# Arguments:
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type Noise <: Kernel
    σ2::Float64      # Log of Signal std
    Noise(lσ::Float64) = new(exp(2.0*lσ))
end

cov(noise::Noise, sameloc::Bool) = noise.σ2*sameloc
function cov(noise::Noise, x::Vector{Float64}, y::Vector{Float64})
    return cov(noise, (norm(x-y)<eps()))
end

get_params(noise::Noise) = Float64[log(noise.σ2)/2.0]
get_param_names(noise::Noise) = [:lσ]
num_params(noise::Noise) = 1

function set_params!(noise::Noise, hyp::Vector{Float64})
    length(hyp) == 1 || throw(ArgumentError("Noise kernel only has one parameter"))
    noise.σ2 = exp(2.0*hyp[1])
end

@inline dk_dlσ(noise::Noise, sameloc=Bool) = 2.0*cov(noise,sameloc)
@inline function dKij_dθp(noise::Noise, X::Matrix{Float64}, i::Int, j::Int, p::Int, dim::Int)
    return dk_dlσ(noise, norm(X[:,i]-X[:,j])<eps())
end
@inline function dKij_dθp(noise::Noise, X::Matrix{Float64}, data::EmptyData, i::Int, j::Int, p::Int, dim::Int)
    return dKij_dθp(noise, X, i, j, p, dim)
end
