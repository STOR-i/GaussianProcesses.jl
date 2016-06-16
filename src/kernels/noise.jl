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

function cov(noise::Noise, x::Vector{Float64}, y::Vector{Float64})
    sigma2 = exp(2*noise.lσ)
    prec   = eps()            #machine precision
    
    K =  sigma2*(norm(x-y)<prec)
    return K
end

get_params(noise::Noise) = Float64[noise.lσ]
get_param_names(noise::Noise) = [:lσ]
num_params(noise::Noise) = 1

function set_params!(noise::Noise, hyp::Vector{Float64})
    length(hyp) == 1 || throw(ArgumentError("Noise kernel only has one parameter"))
    noise.lσ = hyp[1]
end

function grad_kern(noise::Noise, x::Vector{Float64}, y::Vector{Float64})
    sigma2 = exp(2*noise.lσ)
    prec   = eps()            #machine precision
    
    dK_sigma = 2.0*sigma2*(norm(x-y)<prec)
    
    dK_theta = [dK_sigma]
    return dK_theta
end
