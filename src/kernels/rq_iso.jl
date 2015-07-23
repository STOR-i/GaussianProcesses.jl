# Rational Quadratic Isotropic Covariance Function 

@doc """
# Description
Constructor for the isotropic Rational Quadratic kernel (covariance)

k(x,x') = σ²(1+(x-x')ᵀ(x-x')/2αℓ²)^{-α}
# Arguments:
* `ll::Float64`: Log of length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
* `lα::Float64`: Log of shape parameter α
""" ->
type RQIso <: Kernel
    ll::Float64      # Log of length scale 
    lσ::Float64      # Log of signal std
    lα::Float64      # Log of shape parameter
    RQIso(ll::Float64, lσ::Float64, lα::Float64) = new(ll, lσ, lα)
end

function kern(rq::RQIso, x::Vector{Float64}, y::Vector{Float64})
    ℓ2 = exp(2.0*rq.ll)
    σ2 = exp(2.0*rq.lσ)
    α = exp(rq.lα)
    K =  σ2*(1.0+sqeuclidean(x, y)/(2.0*α*ℓ2)).^(-α)
    return K
end

get_params(rq::RQIso) = Float64[rq.ll, rq.lσ, rq.lα]
num_params(rq::RQIso) = 3

function set_params!(rq::RQIso, hyp::Vector{Float64})
    length(hyp) == 3 || throw(ArgumentError("Rational Quadratic function has three parameters"))
    rq.ll, rq.lσ, rq.lα = hyp
end

function grad_kern(rq::RQIso, x::Vector{Float64}, y::Vector{Float64})
    ℓ2 = exp(2.0*rq.ll)
    σ2 = exp(2.0*rq.lσ)
    α  = exp(rq.lα)
    dxy2 = sqeuclidean(x,y)
    
    dK_dℓ = σ2*((dxy2)/ℓ2)*(1.0+(dxy2)/(2*α*ℓ2))^(-α-1.0) # Is this correct?
    dK_dσ = 2.0*σ2*(1+(dxy2)/(2*α*ℓ2))^(-α)
    
    part = (1.0+dxy2/(2*α*ℓ2))
    dK_dα = σ2*part^(-α)*((dxy2)/(2*ℓ2*part)-α*log(part))
    dK_dθ = [dK_dℓ, dK_dσ, dK_dα]
    return dK_dθ
end

function crossKern(X::Matrix{Float64}, rq::RQIso)
    ℓ2 = exp(2.0*rq.ll)
    σ2 = exp(2.0*rq.lσ)
    α = exp(rq.lα)

    R = pairwise(SqEuclidean(), X)
    broadcast!(/, R, R, 2.0*α*ℓ2)
    broadcast!(+, R, R, 1.0)
    broadcast!(^, R, R, -α)
    broadcast!(*, R, R, σ2)
end

function grad_stack!(stack::AbstractArray, X::Matrix{Float64}, rq::RQIso)
    d, nobsv = size(X)
    ℓ2 = exp(2.0 * rq.ll)
    σ2 = exp(2.0 * rq.lσ)
    α = exp(rq.lα)
    dxy2 = pairwise(SqEuclidean(), X)
    
    for i in 1:nobsv, j in 1:i
        # Check these derivatives!
        @inbounds stack[i,j,1] = σ2*((dxy2[i,j])/ℓ2)*(1.0+(dxy2[i,j])/(2*α*ℓ2))^(-α-1.0)  # dK_dℓ
        @inbounds stack[j,i,1] = stack[i,j,1]
        @inbounds stack[i,j,2] = 2.0*σ2*(1+(dxy2[i,j])/(2*α*ℓ2))^(-α)    # dK_dσ
        @inbounds stack[j,i,2] = stack[i,j,2]
        part = (1.0+dxy2[i,j]/(2*α*ℓ2))
        @inbounds stack[i,j,3] = σ2*part^(-α)*((dxy2[i,j])/(2*ℓ2*part)-α*log(part))  # dK_dα
        @inbounds stack[j,i,3] = stack[i,j,3]        
    end
    return stack
end
