#White Noise kernel

@doc """
# Description
Constructor for the Noise kernel (covariance)

k(x,x') = σ²δ(x-x'), where δ is a Kronecker delta function and equals 1 iff x=x' and zero otherwise
# Arguments:
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type Noise <: Kernel
    lσ::Float64      # Log of Signal std
    Noise(lσ::Float64) = new(lσ)
end

function kern(noise::Noise, x::Vector{Float64}, y::Vector{Float64})
    sigma2 = exp(2*noise.lσ)
    prec   = eps()            #machine precision
    
    K =  Float64[norm(x-y)<prec]
    return K
end

get_params(noise::Noiseo) = Float64[noise.lσ]
num_params(noise::Noise) = 1

function set_params!(noise::Noise, hyp::Vector{Float64})
    length(hyp) == 1 || throw(ArgumentError("Noise kernel only has one parameter"))
    noise.lσ = hyp
end

function grad_kern(noise::Noise, x::Vector{Float64}, y::Vector{Float64})
    sigma2 = exp(2*noise.lσ)
    
    dK_sigma = 2.0*sigma2*
    
    dK_theta = [dK_sigma]
    return dK_theta
end
